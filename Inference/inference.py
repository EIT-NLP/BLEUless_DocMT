import os
import json
import time
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from evaluator.eval_utils import get_eval_metrics
from tqdm import tqdm
from transformers import set_seed

# Code adapted from https://github.com/wxjiao/ParroT
# Run Example
# Python3 inference.py --model_name_or_path wxjiao/alpaca-7b --hf_cache_dir /home/ec2-user/hf_cache
# --input_file /efs/daweizhu/data/remove_me/test_rand_50.de.txt
# --output_file /efs/daweizhu/data/remove_me/test_rand_50.de-en.txt
# --lang_pair de-en --inst_file /efs/daweizhu/data/remove_me/instruct_inf.txt
# --reference_file /efs/daweizhu/data/remove_me/test_rand_50.en.txt


# Instruction language, default: 'en'
lang_instruction = {
    'de': {'de': "Deutsch", 'en': "Englisch", 'ja': "Japanisch", 'zh': "Chinesisch"},
    'en': {'de': "German", 'en': "English", 'ja': "Japanese", 'zh': "Chinese",
           'cs': "Czech", 'ha': "Hausa", 'ru': "Russian", 'is': "Icelandic", 'uk': "Ukrainian",
           'tr': "Turkish", 'fr': "French", 'it': "Italian", 'es': "Spanish", 'pt': "Portuguese", 'nl': "Dutch",
           'pl': "Polish", 'ro': "Romanian", 'bg': "Bulgarian", 'el': "Greek", 'fi': "Finnish", 'sv': "Swedish",
           'da': "Danish", 'no': "Norwegian", 'hu': "Hungarian", 'et': "Estonian", 'lv': "Latvian",
           'lt': "Lithuanian", 'hr': "Croatian", 'sr': "Serbian", 'sl': "Slovenian", 'sk': "Slovak",
           'sq': "Albanian", 'mk': "Macedonian", 'hy': "Armenian", 'ka': "Georgian", 'he': "Hebrew",
           'ar': "Arabic", 'ur': "Urdu", 'fa': "Persian", 'hi': "Hindi", 'bn': "Bengali", 'th': "Thai",
           'lo': "Lao", 'my': "Burmese", 'km': "Khmer", 'vi': "Vietnamese", 'id': "Indonesian", 'ms': "Malay"},
    'ja': {'de': "ドイツ語", 'en': "英語", 'ja': "日本語", 'zh': "中国語"},
    'zh': {'de': "德语", 'en': "英语", 'ja': "日语", 'zh': "中文"},
}

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# Special tokens in llama
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


# Read task instruction, fill in languages
def read_instruct(path, src, tgt, lang_ins="en"):
    source, target = lang_instruction[lang_ins][src], lang_instruction[lang_ins][tgt]
    ins_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            line = l.strip().replace("[SRC]", source).replace("[TGT]", target)
            ins_list.append(line)
    return ins_list


# Read input data for inference
def read_jsonl_input(path):
    input_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            input_data.append(json.loads(l.strip()))
    return input_data

def read_txt_input(path):
    input_data = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            input_data.append(l.strip())
    return input_data

def add_bloom_mt_prompt(tokenizer, source_language_full_name, target_language_full_name, input_data):
    # MT tuned llama
    prefix = f"Translate the following text from {source_language_full_name} to {target_language_full_name}."

    list_data_dict = [{"prefix": prefix, "input": p.strip(),
                       "source_language_full_name": source_language_full_name,
                       "target_language_full_name": target_language_full_name} for p in input_data]
    prompt = "{prefix}\n{source_language_full_name}:{input}\n{target_language_full_name}:"
    sources = [prompt.format_map(example) for example in list_data_dict]
    return sources


def remove_bloom_mt_prompt(model_name, text, source_language_full_name, target_language_full_name):
    text = text.split(f"\n{target_language_full_name}:")[-1].strip()
    return text

def add_mistral_mt_prompt(tokenizer, source_language_full_name, target_language_full_name, input_data):
    # Template example: <s>[INST] What is your favourite condiment? [/INST]
    # Attention: if you use tokenizer.apply_chat_template(messages, return_tensors="pt") as suggested
    # in the official page, then you should append the <s>. Because tokenizer.apply_chat_template() will not
    # add special tokens.
    # In our code, tokenizer() will add special tokens, so we don't need to append <s>
    prefix = f"[INST] Translate the following text from {source_language_full_name} to {target_language_full_name}.\n"
    suffix = f" [/INST]"  # We need a space before [/INST], See huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    list_data_dict = [{"prefix": prefix, "input": p.strip(), "suffix": suffix,
                       "source_language_full_name": source_language_full_name,
                       "target_language_full_name": target_language_full_name} for p in input_data]
    prompt = "{prefix}{input}{suffix}"
    sources = [prompt.format_map(example) for example in list_data_dict]

    # sanity check
    sanity_tokens = tokenizer(sources[0])
    input_ids_sanity_tokens = sanity_tokens["input_ids"]
    assert input_ids_sanity_tokens[0] == 1 and input_ids_sanity_tokens[1] != 1

    return sources


def remove_mistral_mt_prompt(model_name, text, source_language_full_name, target_language_full_name):
    text = text.split(f" [/INST]")[-1].strip()
    return text


def add_llama_mt_prompt(tokenizer, source_language_full_name, target_language_full_name, input_data):
    # MT tuned llama
    prefix = f"Translate the following text from {source_language_full_name} to {target_language_full_name}. {source_language_full_name}:"

    list_data_dict = [{"prefix": prefix, "input": p.strip(),
                       "source_language_full_name": source_language_full_name,
                       "target_language_full_name": target_language_full_name} for p in input_data]
    prompt = "{prefix} {input} {target_language_full_name}:"
    sources = [prompt.format_map(example) for example in list_data_dict]
    return sources


def remove_llama_mt_prompt(model_name, text, source_language_full_name, target_language_full_name):
    text = text.split(f"{target_language_full_name}: ")[-1].strip()
    return text


def add_vanilla_prompt(tokenizer, source_language_full_name, target_language_full_name, input_data):
    # Vanilla prompt
    prompt_format = "{source_language_full_name}: {source_text}\n{target_language_full_name}:"

    sources = [prompt_format.format(
        source_language_full_name=source_language_full_name,
        source_text=p.strip(),
        target_language_full_name=target_language_full_name) for p in input_data]

    return sources


def remove_vanilla_prompt(model_name, text, source_language_full_name, target_language_full_name):
    # Remove vanilla prompt
    try:
        _, translated_text = text.split(f"{target_language_full_name}:", 1)
    except ValueError:
        return text  # Return original text if the split fails
    return translated_text.strip()



def add_llama_zero_shot_prompt(tokenizer, source_language_full_name, target_language_full_name, input_data):
    # original open llama

    list_data_dict = [{"input": p.strip(),
                       "source_language_full_name": source_language_full_name,
                       "target_language_full_name": target_language_full_name} for p in input_data]
    prompt = "Translate the following {source_language_full_name} text in {target_language_full_name}. {source_language_full_name} text: {input} {target_language_full_name} text: "
    sources = [prompt.format_map(example) for example in list_data_dict]
    return sources


def remove_llama_zero_shot_prompt(model_name, text, source_language_full_name, target_language_full_name):
    text = text.split(f"{target_language_full_name} text:")[-1].strip()
    return text


def add_zephyr_prompt(tokenizer, source_language_full_name, target_language_full_name, input_data):
    # instruction = f"Translate the following text from {source_language_full_name} to {target_language_full_name}."
    input_with_prompt = [f"{p.strip()}" for p in input_data]

    msg_list = []
    for input_p in input_with_prompt:
        messages = [
            {
                "role": "system",
                "content": f"You are a translation agent and you should translate the user's input text from {source_language_full_name} to {target_language_full_name}",
            },
            {"role": "user", "content": input_p},
        ]
        msg_list.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


    return msg_list


def remove_zephyr_prompt(model_name, text, source_language_full_name, target_language_full_name):
    text = text.split(f"<|assistant|>")[-1].strip()
    return text


def add_vicuna_prompt(tokenizer, source_language_full_name, target_language_full_name, input_data):
    # "USER: {instruction} {source_language_full_name}: {input} ASSISTANT:"
    instruction = f"Translate the following text from {source_language_full_name} to {target_language_full_name}."
    list_data_dict = [{"instruction": instruction, "input": p.strip(),
                       "source_language_full_name": source_language_full_name} for p in input_data]

    prompt = "USER: {instruction}\n{input}\nASSISTANT:"
    sources = [prompt.format_map(example) for example in list_data_dict]
    return sources


def remove_vicuna_prompt(model_name, text, source_language_full_name, target_language_full_name):
    text = text.split(f"ASSISTANT:")[-1].strip()
    return text


def add_alpaca_prompt(tokenizer, source_language_full_name, target_language_full_name, input_data):
    instruction = f"Translate from {source_language_full_name} to {target_language_full_name}."
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. " \
             "Write a response that appropriately completes the request.\n\n" \
             "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"

    list_data_dict = [{"instruction": instruction, "input": p.strip()} for p in input_data]

    sources = [prompt.format_map(example) for example in list_data_dict]
    return sources


def remove_alpaca_prompt(model_name, text, source_language_full_name, target_language_full_name):
    text = text.split("### Response:")[1].strip()
    text = text.replace("\n", " ")
    # Cut for contrastive instruction
    if "</p>" in text:
        text = text.split("</p>")[0].split("<p>")[-1]

    return text


def get_prompt_functions(args):
    model_name = args.model_name_or_path.lower()
    add_prompt, remove_prompt = None, None

    prompt_types = {
        "alpaca": (add_alpaca_prompt, remove_alpaca_prompt),
        "bloomz": (add_bloom_mt_prompt, remove_bloom_mt_prompt),
        "vicuna": (add_vicuna_prompt, remove_vicuna_prompt),
        "mistral": (add_mistral_mt_prompt, remove_mistral_mt_prompt),
        "llama": (add_llama_mt_prompt, remove_llama_mt_prompt),
        "vanilla": (add_vanilla_prompt, remove_vanilla_prompt)
    }

    # Fetch prompt functions based on user-specified prompt type
    add_prompt, remove_prompt = prompt_types.get(args.prompt_type)
    if add_prompt is None or remove_prompt is None:
        raise NotImplementedError("Unknown template type")

    return add_prompt, remove_prompt


# Assembly instruction and input data, handle hints
def create_prompt(args, tokenizer, model_name, input_data, source_lang,
                  target_lang, add_prompt_fn):
    source_language_full_name = lang_instruction['en'][source_lang]
    target_language_full_name = lang_instruction['en'][target_lang]

    source_sentences = add_prompt_fn(tokenizer, source_language_full_name, target_language_full_name, input_data)

    return source_sentences


# Post-process the output, extract translations
def post_process(args, model_name, text, source_lang, target_lang, remove_prompt_fn):
    source_language_full_name = lang_instruction['en'][source_lang]
    target_language_full_name = lang_instruction['en'][target_lang]
    pure_text = remove_prompt_fn(model_name, text, source_language_full_name, target_language_full_name)
    pure_text = pure_text.strip()
    pure_text = pure_text.replace("\n", " ")

    return pure_text


def is_bloom_family(model_name_or_path):
    return "bloom" in model_name_or_path


def is_vicuna_family(model_name_or_path):
    return "vicuna" in model_name_or_path


def is_zephyr_family(model_name_or_path):
    return "zephyr" in model_name_or_path


def is_alpaca_family(model_name_or_path):
    return any(keyword in model_name_or_path for keyword in ["alpaca", "alpapig"])


def is_falcon_family(model_name_or_path):
    return "falcon" in model_name_or_path.lower()


def supported_decoder_only_model(model_name_or_path):
    # List of specific model names
    decoder_only_models = [
        "gpt2", "openlm-research/open_llama_7b", "wxjiao/ParroT-7b",
        "wxjiao/alpaca-7b", "wxjiao/ParroT-Hint-7b", "lmsys/vicuna-7b-v1.5",
        "lmsys/vicuna-7b-v1.3", "dpml/in-house-alpaca", "dpml/a_mono"
    ]

    # Additional keywords to check in the model name
    additional_keywords = ["dpo_mt", "pro_mt", "llama", "baseline", "checkpoint", "mistral", "preRT", "l1a"]

    # Check if the model name or path is in the list or matches any family or keyword
    return (
        model_name_or_path in decoder_only_models or
        is_vicuna_family(model_name_or_path) or
        is_bloom_family(model_name_or_path) or
        is_falcon_family(model_name_or_path) or
        is_zephyr_family(model_name_or_path) or
        is_alpaca_family(model_name_or_path) or
        any(keyword in model_name_or_path for keyword in additional_keywords) or
        "baseline" in model_name_or_path and "checkpoint" in model_name_or_path or
        "mistral" in model_name_or_path.lower()
    )




def convert_flores_50_to_language_code(lang):
    flores_lang_map = {"ces_Latn": "cs", "deu_Latn": "de", "eng_Latn": "en", "spa_Latn": "es", "fra_Latn": "fr",
                       "ita_Latn": "it", "nld_Latn": "nl", "por_Latn": "pt", "ron_Latn": "ro", "rus_Cyrl": "ru",
                       "swe_Latn": "sv", "tur_Latn": "tr", "vie_Latn": "vi", "zho_Hans": "zh", "hau_Latn": "ha",
                       "isl_Latn": "is", "ind_Latn": "id", "jpn_Latn": "ja", "kor_Latn": "ko", "mar_Latn": "mr"}
    if lang in flores_lang_map:
        return flores_lang_map[lang]
    else:
        return lang

# def reference_file_name_mapping(reference_file, lang_pair, output_suffix):
#     # if reference file exists, then just use it
#     if os.path.exists(reference_file):
#         return reference_file
#
#     if output_suffix == "wmt21":
#         reference_file = reference_file.replace(".ref", ".ref-A")
#     elif output_suffix == "wmt22":
#         if lang_pair in ["cs-en", "en-cs"]:
#             reference_file = reference_file.replace(".ref", ".ref-B")
#         elif lang_pair in ["liv-en", "ru-sah"]:
#             pass
#         else:
#             reference_file = reference_file.replace(".ref", ".ref-A")
#
#     return reference_file


def sort_prompt_by_length(prompts, tokenizer):
    # By sorting the prompts by length, we can reduce the number of padding tokens, and thus speed up the decoding
    # Empirically, I find that sorting the prompts by length can speed up the decoding by 2x
    # Generate prompt input IDs and calculate lengths
    prompt_lengths = [len(tokenizer(ex, padding=False, truncation=True, return_tensors="pt")["input_ids"][0])
                      for ex in prompts]

    # Sort the indices based on prompt lengths
    sorted_prompt_idx = np.argsort(prompt_lengths)

    # Sort prompts based on the sorted indices
    sorted_prompts = [prompts[idx] for idx in sorted_prompt_idx]

    return sorted_prompts, sorted_prompt_idx


def initialize_tokenizer(model_name_or_path, model_cache_dir):
    use_fast = "bloom" in model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=use_fast,
        cache_dir=model_cache_dir
    )
    return tokenizer

def setup_model(args, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        cache_dir=args.hf_cache_dir,
        device_map="auto"
    )
    # print(model.hf_device_map)
    return model


def get_generation_config(args, tokenizer, search_type):
    gen_config_args = {
        'temperature': args.temperature,
        'top_p': 0.9,
        'num_beams': 4 if search_type == "beam" else 1,
        'max_new_tokens': 512,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token': tokenizer.pad_token_id
    }
    return GenerationConfig(**gen_config_args, do_sample=True) if search_type == "greedy" else GenerationConfig(**gen_config_args)

def convert_model_name_to_str(model_name):
    # basically, convert all '/' to '-', but if there are more than 4 '/'s, only keep the last 4

    model_name_split = model_name.split("/")
    if len(model_name_split) > 5:
        model_name_split = model_name_split[-5:]

    return "-".join(model_name_split)


def main():
    parser = argparse.ArgumentParser()

    # Training settings

    # Model Settings
    parser.add_argument('--model_name_or_path', type=str, default='gpt2')
    parser.add_argument('--hf_cache_dir', type=str, required=True, help='cache dir for HF models')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--precision', choices=["fp16", "fp32", "bf16"], default="fp16")
    parser.add_argument('--use_fast_tokenizer', choices=["true", "false"], default="true")
    parser.add_argument('--generation_max_length', type=int, default=64)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--search', choices=["greedy", "beam"], default="beam")
    parser.add_argument('--prompt_type', choices=["alpaca", "vicuna", "mistral", "bloomz", "llama", "vanilla"], default="alpaca")
    parser.add_argument('--eval_metric', choices=["offline", "comet"], default="comet")

    # dataset settings
    parser.add_argument('--input_file_path', type=str, required=True, help='input file')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory')
    parser.add_argument('--output_suffix', type=str, required=True, help='output suffix')

    parser.add_argument('--lang_pair', type=str, default='zh-en', help='language pair: zh-en, en-de')

    # torch 2.0 settings
    parser.add_argument('--torch_compile', choices=["true", "false"], default="false")

    # deepspeed related
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Enable deepspeed and pass the path to deepspeed json config file")

    args = parser.parse_args()

    start = time.time()
    set_seed(42)

    # Extract and assert language pair and file names
    lang_pair = args.lang_pair
    srcl, tgtl = lang_pair.split('-')
    input_file_name = args.input_file_path.split("/")[-1].replace("txt", "")
    assert lang_pair in input_file_name

    model_name_or_path_str = convert_model_name_to_str(args.model_name_or_path)

    output_dir = os.path.join(args.output_dir,
                              f"{args.output_suffix}-{model_name_or_path_str}-{args.prompt_type}-{args.search}-{args.precision}",
                              f"test.{lang_pair}")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set precision and check for validity
    precision_to_dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    torch_dtype = precision_to_dtype.get(args.precision)
    if torch_dtype is None:
        raise NotImplementedError("Unsupported Precision")

    # Additional arguments
    bs = args.per_device_eval_batch_size

    tokenizer = initialize_tokenizer(args.model_name_or_path, args.hf_cache_dir)
    model = setup_model(args, torch_dtype)

    # Set padding side for supported models
    if supported_decoder_only_model(args.model_name_or_path):
        tokenizer.padding_side = "left"
    else:
        raise NotImplementedError("Unsupported Model")

    # Ensure tokenizer has pad_token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup generation configuration
    gen_config = get_generation_config(args, tokenizer, args.search)

    # read the input and add instructions to the input
    input_data = read_jsonl_input(args.input_file_path)
    source_sentences = [d['input'] for d in input_data]
    add_prompt_function, remove_prompt_function = get_prompt_functions(args)
    prompt = create_prompt(args, tokenizer, args.model_name_or_path, source_sentences, srcl, tgtl, add_prompt_function)
    prompt, sort_idx = sort_prompt_by_length(prompt, tokenizer)

    # Define output file path
    output_file = os.path.join(output_dir, f"prediction_{lang_pair}.txt")

    # Generate
    decoded_tokens = []
    with open(output_file, 'w', encoding='utf-8') as fo, open(output_file + ".hyp", 'w', encoding='utf-8') as fo2:
        for i in tqdm(range(0, len(prompt), bs), desc=f"inferring...tag: [{args.output_suffix}], lang: [{lang_pair}]"):
            p = prompt[i:i + bs]
            tokenized = tokenizer(p, padding=True, return_tensors="pt")
            input_ids = tokenized.input_ids.cuda()
            attn_mask = tokenized.attention_mask.cuda()
            with torch.no_grad():
                generated_ids = model.generate(inputs=input_ids, attention_mask=attn_mask, generation_config=gen_config).cpu().detach()
            decoded_tokens_batch = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            decoded_tokens.extend(decoded_tokens_batch)
        assert len(decoded_tokens) == len(prompt)
        # sort the decoded tokens back to the original order
        recovered_order_decoded_tokens = [None] * len(prompt)
        for i, idx in enumerate(sort_idx):
            recovered_order_decoded_tokens[idx] = decoded_tokens[i]
        assert None not in recovered_order_decoded_tokens
        for dec in recovered_order_decoded_tokens:
            print(dec, file=fo, flush=True)
            print(post_process(args, args.model_name_or_path, dec, srcl, tgtl, remove_prompt_function), file=fo2, flush=True)

    print(f"output files written, now computing the scores")

    source_text = source_sentences
    pred_text = read_txt_input(f"{output_file}.hyp")
    reference_text = [d['output'] for d in input_data]
    metric_evaluator = get_eval_metrics(args.eval_metric)
    if args.eval_metric == "comet":
        metric_result = metric_evaluator.gpu_compute(pred_text, reference_text, source_text)
    else:
        metric_result = metric_evaluator.compute(pred_text, reference_text, source_text)
        
    ''' 
    1. set args.eval_metric == "d-BLEU"
    2. use metric_evaluator.compute for d-BLEU and ave_compute for AvgBLEU
    3. tips: the input for compute and ave_compute must be the document-level translation combined according to the boundary    
    4. please change the prompt when evaluting different metrics and models
    '''    
        
        
    print(f"******* {args.eval_metric} SCORE [{srcl}-{tgtl}]*******")
    print(metric_result)

    results_output_file = os.path.join(output_dir, f"prediction_{lang_pair}_score.txt")
    with open(results_output_file, 'w', encoding='utf-8') as fo:
        json.dump(metric_result, fo)

    end = time.time()
    print(f"results written to {results_output_file}")
    print(f"Total time taken for inference: {end - start} seconds")
    print(end - start)


if __name__ == '__main__':
    main()
