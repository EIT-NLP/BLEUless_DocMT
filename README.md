# BLEUless_DocMT
# Instruction-Tuned LLMs Succeed in Document-Level MT Without Fine-Tuning—But BLEU Turns a Blind Eye

## Introduction

This repository contains the code and data for our paper, **"Instruction-Tuned LLMs Succeed in Document-Level MT Without Fine-Tuning—But BLEU Turns a Blind Eye"**. Our work explores the ability of instruction-tuned large language models (LLMs) to handle document-level machine translation (docMT) without requiring specialized document-level training. We assess whether instruction-tuned LLMs can translate entire documents in a single pass, achieving coherent and context-aware translations beyond sentence-level methods.

In contrast to prior studies focusing on sentence-by-sentence translation, we demonstrate that LLMs prompted to translate entire documents at once deliver higher-quality outputs, preserving document-level context and improving coherence. However, traditional n-gram metrics like BLEU fail to reflect this advantage, often favoring sentence-based translations. To address this evaluation gap, we propose an **LLM-as-a-judge** paradigm, where GPT-4 assesses translations based on coherence, accuracy, and fluency, offering a more nuanced and human-like evaluation.

### Key Contributions

- **LLM-as-a-Judge Paradigm**: We design tailored prompts for GPT-4 to assess document-level translation, capturing aspects of fluency, coherence, and accuracy that traditional metrics overlook.
- **Entire Document Translation V.S. Sentence-merged Translation**: Our experiments show that translating entire documents in one pass yields more coherent and accurate results than independent sentences translations and then merged, even without fine-tuning for docMT.
- **Evaluation Insights**: We recommend against using BLEU scores for docMT, as they fail to capture discourse-level coherence and can often produce misleading results, particularly in document-level evaluations.

Code and data are coming soon.
