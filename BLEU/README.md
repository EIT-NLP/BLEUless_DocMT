# BLEU-based Evaluation

Datasets and Experimental Results in [3 BLEU-based Evaluation].

### **Dataset Details**

We use translation benchmarks with document boundary: WMT22(zh-en, en-zh, de-en, en-de)

<img src="BLEU/data_stat.png" alt="image-20241119145854530" style="zoom: 50%;" />

### Translation Approaches

Given a document containing \( l \) source sentences  
\( X = \{x^1, \dots, x^l\} \), the goal of docMT is to generate its translation  
\( Y = \{y^1, \dots, y^l\} \) as a sequence of sentences in the target language.  
In this work, we explore two approaches for generating translations using instruction-tuned LLMs:

- **ST[k]**: We concatenate \( k \) source sentences into a chunk, input each chunk into the LLM for translation, and then concatenate the translated chunks together to form the full document translation.

- **DOC**: We instruct the LLM to directly translate the entire document in one pass.

### Evaluation Metrics

We argue that documents are generally independent units, so they should be weighted equally in the evaluation. We, therefore, propose an alternative, AvgBLEU, defined as:

\[
\text{AvgBLEU} = \frac{1}{N} \sum_{i=1}^{N} \text{BLEU}\left(Y_i^{\text{ref}}, Y_i^{\text{pred}}\right)
\]

Here, \( N \) is the number of documents, and \( Y^{\text{ref}} \) and \( Y^{\text{pred}} \) represent the reference document translations and the predicted translations, respectively. This allows us to calculate the average BLEU score (AvgBLEU) for the entire dataset, providing a comprehensive measure of translation quality. For completeness, we report results using the standard d-BLEU in [Appendix B d-BLEU Performance].

![image-20241119151858660](BLEU/AvgBLEU.png)

<img src="BLEU/AvgBLEU.png/d-BLEU.png" alt="image-20241119151948194" style="zoom:55%;" />
