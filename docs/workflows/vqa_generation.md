# VQA Generation

The Visual Question Answering (VQA) generation workflow creates question-answer pairs from a set of source documents. This is essential for training models that can understand and answer questions about the content and layout of a document.

A key feature of this workflow is its ability to generate **hard negatives**—answers that are contextually similar but incorrect—which helps in training more robust retrieval and ranking models.

## Usage

You can generate a VQA dataset from documents you've just created or from an existing set of images.

```python
from synthdoc import SynthDoc

# Initialize SynthDoc
synth = SynthDoc()

# First, generate some source documents
source_dataset = synth.generate_raw_docs(
    language="en",
    num_pages=1,
    prompt="Create a one-page summary of a company's quarterly earnings report."
)

# Now, generate VQA pairs from these documents
vqa_dataset = synth.generate_vqa(
    source_documents=source_dataset,
    num_questions_per_doc=5
)

# The output is a HuggingFace Dataset
print(vqa_dataset)
print("Generated Questions:", vqa_dataset[0]['questions'])
print("Generated Answers:", vqa_dataset[0]['answers'])
```

## Parameters

-   `source_documents` (Dataset): A HuggingFace `Dataset` containing the documents to process. This is typically the output of `generate_raw_docs`.
-   `num_questions_per_doc` (int): The number of question-answer pairs to generate for each document.
-   `hard_negative_ratio` (float): The ratio of hard negative examples to include. Defaults to `0.3`.

## Output

This method extends the input `Dataset` with additional columns for VQA:

-   `questions`: A list of generated questions.
-   `answers`: A list of corresponding ground truth answers.
-   `hard_negatives`: A list of challenging, incorrect answers.
-   Other metadata related to the questions and their types.
