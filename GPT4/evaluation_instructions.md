## **Prompt for Fluency Evaluation**

Please evaluate the fluency of the following text in the target language (English, Chinese, or German).

------

### **Instructions:**

- **Task**: Evaluate the fluency of the text.

- Scoring: Provide a score from 1 to 5, where:

  - **5**: The text is **highly fluent**, with no grammatical errors, unnatural wording, or stiff syntax.
  - **4**: The text is **mostly fluent**, with minor errors that do not impede understanding.
  - **3**: The text is **moderately fluent**, with noticeable errors that may slightly affect comprehension.
  - **2**: The text has **low fluency**, with frequent errors that hinder understanding.
  - **1**: The text is **not fluent**, with severe errors that make it difficult to understand.
- **Explanation**: Support your score with specific examples to justify your evaluation.

------

### **Output Format:**

Provide your evaluation in the following JSON format:

```
{
  "Fluency": {
    "Score": "<the score>",
    "Explanation": "<your explanation on how you made the decision>"
  }
}
```

------

**Text to Evaluate:**

[Insert the text here]

------

------

## **Prompt for Accuracy Evaluation**

Please evaluate the accuracy of the following text by comparing it to the reference text provided.

------

### **Instructions:**

- **Task**: Compare the text to the reference text.

- Identify Mistakes: List all mistakes related to accuracy.

  - Mistake Types:

    - **Wrong Translation**: Incorrect meaning or misinterpretation leading to wrong information.
    - **Omission**: Missing words, phrases, or information present in the reference text.
    - **Addition**: Extra words, phrases, or information not present in the reference text.
    - **Others**: Mistakes that are hard to define or categorize.

- **Note**: If the text expresses the same information as the reference text but uses different words or phrasing, it is **not** considered a mistake.

- **Provide a List**: Summarize all mistakes without repeating the exact sentences. Provide an empty list if there are no mistakes.

------

### **Output Format:**

Provide your evaluation in the following JSON format:

```
{
  "Accuracy": {
    "Mistakes": [
      "<list of all mistakes in the text, provide an empty list if there are no mistakes>"
    ]
  }
}
```

------

**Reference Text:**

[Insert the reference text here]

**Text to Evaluate:**

[Insert the text here]

------

------

## **Prompt for Cohesion Evaluation**

Please evaluate the cohesion of the following text by comparing it to the reference text.

------

### **Instructions:**

- **Task**: Evaluate the cohesion of the text.

- **Definition**: Cohesion refers to how different parts of a text are connected using language structures like grammar and vocabulary. It ensures that sentences flow smoothly and the text makes sense as a whole.

- Identify Mistakes: List all mistakes related to cohesion.

  - Separate the mistakes into:

    - **Lexical Cohesion Mistakes**: Issues with vocabulary usage, incorrect or missing synonyms, or overuse of certain words that disrupt the flow.
    - **Grammatical Cohesion Mistakes**: Problems with pronouns, conjunctions, or grammatical structures that link sentences and clauses.

- **Provide Lists**: Provide separate lists for lexical cohesion mistakes and grammatical cohesion mistakes. Provide empty lists if there are no mistakes.

------

### **Output Format:**

Provide your evaluation in the following JSON format:

```
{
  "Cohesion": {
    "Lexical Cohesion Mistakes": [
      "<list of all mistakes in the text, provide an empty list if there are no mistakes>"
    ],
    "Grammatical Cohesion Mistakes": [
      "<list of all mistakes in the text, provide an empty list if there are no mistakes>"
    ]
  }
}
```

------

**Reference Text:**

[Insert the reference text here]

**Text to Evaluate:**

[Insert the text here]

------

