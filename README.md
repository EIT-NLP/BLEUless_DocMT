<div align="center">

# Instruction-Tuned LLMs Succeed in Document-Level MT Without Fine-Tuning—But BLEU Turns a Blind Eye

### Resources&Datasets
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/pdf/2410.20941) [![WMT22](https://img.shields.io/badge/Dataset-WMT22-blue)](https://aclanthology.org/2022.wmt-1.1/)

### Models
[![Vicuna-7B](https://img.shields.io/badge/Model-Vicuna--7B-21C2A4)](https://huggingface.co/lmsys/vicuna-7b-v1.5)  [![Vicuna-7B-16K](https://img.shields.io/badge/Model-Vicuna--7B--16K-21C2A4)](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k)  [![Vicuna-13B](https://img.shields.io/badge/Model-Vicuna--13B-21C2A4)](https://huggingface.co/lmsys/vicuna-13b-v1.5)  [![Vicuna-13B-16K](https://img.shields.io/badge/Model-Vicuna--13B--16K-21C2A4)](https://huggingface.co/lmsys/vicuna-13b-v1.5-16k)  [![Mistral-7B](https://img.shields.io/badge/Model-Mistral--7B-21C2A4)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)

</div>


## 1. Introduction

This repository contains the code and data for our paper, **"Instruction-Tuned LLMs Succeed in Document-Level MT Without Fine-Tuning—But BLEU Turns a Blind Eye"**. Our work explores the ability of instruction-tuned large language models (LLMs) to handle document-level machine translation (docMT) without requiring specialized document-level training. We assess whether instruction-tuned LLMs can translate entire documents in a single pass, achieving coherent and context-aware translations beyond sentence-level methods.

In contrast to prior studies focusing on sentence-by-sentence translation, we demonstrate that LLMs prompted to translate entire documents at once deliver higher-quality outputs, preserving document-level context and improving coherence. However, traditional n-gram metrics like BLEU fail to reflect this advantage, often favoring sentence-based translations. To address this evaluation gap, we propose an **LLM-as-a-judge** paradigm, where GPT-4 assesses translations based on coherence, accuracy, and fluency, offering a more nuanced and human-like evaluation.

## 2. Key Contributions

- **LLM-as-a-Judge Paradigm**: We design tailored prompts for GPT-4 to assess document-level translation, capturing aspects of fluency, coherence, and accuracy that traditional metrics overlook.
- **Entire Document Translation V.S. Sentence-merged Translation**: Our experiments show that translating entire documents in one pass yields more coherent and accurate results than independent sentences translations and then merged, even without fine-tuning for docMT.
- **Evaluation Insights**: We recommend against using BLEU scores for docMT, as they fail to capture discourse-level coherence and can often produce misleading results, particularly in document-level evaluations.



## 3. Citation

```bibtex
@article{sun2024instruction,
  title={Instruction-Tuned LLMs Succeed in Document-Level MT Without Fine-Tuning--But BLEU Turns a Blind Eye},
  author={Sun, Yirong and Zhu, Dawei and Chen, Yanjun and Xiao, Erjia and Chen, Xinghao and Shen, Xiaoyu},
  journal={arXiv preprint arXiv:2410.20941},
  year={2024}
}
```

## 4. Contact

For questions or collaborations, please contact us at <scxys3@nottingham.edu.cn>.
