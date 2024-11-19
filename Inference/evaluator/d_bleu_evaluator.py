
from sacrebleu.metrics import BLEU

class d_BleuEvaluator(object):
    def __init__(self):
        self.bleu = BLEU()
        
    def compute(self, sys, refs_orin, lang):
        # 根据语言选择合适的分词器
        if lang == "zh":
            lang = 'zh'
        else:
            lang = '13a'
            
       
        refs = [refs_orin]
        
        
        # 使用选择的分词器初始化BLEU实例
        bleu = BLEU(tokenize=lang)
        
        # 使用自定义分词器计算BLEU分数
        temp_scores = bleu.corpus_score(sys, refs)
        
        # 准备详细结果
        result = {
            "bleu": temp_scores.score,
            "precisions": temp_scores.precisions,
            "brevity_penalty": temp_scores.bp,
            "length_ratio": temp_scores.sys_len / temp_scores.ref_len,
            "translation_length": temp_scores.sys_len,
            "reference_length": temp_scores.ref_len
        }
        
        return result
        
     # #-------------各个单独计算版-----------------
 
 
    def ave_compute(self,sys,refs_orin,lang):
        # 根据语言选择合适的分词器
        if lang == "zh":
            lang = 'zh'
        else:
            lang = '13a'
            
        bleu = BLEU(tokenize=lang)

        results = []
        total_score = 0
        for temp_sys, temp_refs in zip (sys, refs_orin):
            individual_result = bleu.corpus_score([temp_sys], [[temp_refs]]).score
            results.append(individual_result)

        for temp_scores in results:
            total_score += temp_scores
        
        ave_score = total_score / len(results)
        
        results.insert(0,f"average: {ave_score}")
        return results
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
# # Load the system translations
# sys_file = "/code/FastChat/David/evaluator/document/jsonl-FastChat-David-huggingface-face-models--lmsys--vicuna-7b-v1.5-vicuna-beam-bf16/test.zh-en/prediction_zh-en.txt.hyp"
# with open(sys_file, "r", encoding="utf-8") as f:
#     sys = f.read().splitlines()
    
# # Load the reference translations
# refs_file = "/code/FastChat/data/inference_dataset_doc/wmt22-zh-en-doc-combined.jsonl"
# refs = []
# with open(refs_file, "r", encoding="utf-8") as f:
#     for line in f:
#         obj = json.loads(line)
#         # Ensure each reference is wrapped in a list
#         refs.append([obj["output"]])
    

# # Load the BLEU metric
# bleu = evaluate.load("bleu")


# results = []  # Store results for each document

# # Compute BLEU scores for each document individually
# for sys_doc, ref_docs in zip(sys, refs):
#     individual_result = bleu.compute(predictions=[sys_doc], references=[ref_docs])
#     results.append(individual_result)

# # Optionally, save the results to a file
# with open("doc_individual_bleu_scores.jsonl", 'w', encoding='utf-8') as f:
#    for doc in results:
#             f.write(json.dumps(doc, ensure_ascii=False) + "\n") 
    
            
# # Print or process the individual results
# for i, result in enumerate(results):
#     print(f"Document {i+1}: BLEU = {result['bleu']:.2f}, Precisions = {result['precisions']}, BP = {result['brevity_penalty']:.2f}, Ratio = {result['length_ratio']:.2f}")
    
