# Document Translation

The document translation workflow translates the text within a document image to different languages while preserving the original layout. This is a powerful tool for creating multi-lingual datasets from a single set of source documents.

## How It Works

The process involves several steps:
1.  **Layout Detection**: A YOLO-based model identifies the locations of text blocks on the page.
2.  **OCR**: The text within each block is extracted.
3.  **Translation**: The extracted text is translated into the target languages.
4.  **Rendering**: The translated text is carefully rendered back onto a copy of the image at the original locations, using fonts appropriate for the target language.

## Usage

To translate documents, use the `translate_documents` method. This workflow requires a YOLO model for layout detection, which will be automatically downloaded on the first run.

```python
from synthdoc import SynthDoc

# Initialize SynthDoc
synth = SynthDoc()

# Generate a source document to translate
source_dataset = synth.generate_raw_docs(language="en", num_pages=1)

# Translate the document into Spanish and French
translated_dataset = synth.translate_documents(
    input_dataset=source_dataset,
    target_languages=["es", "fr"]
)

# The output is a new HuggingFace Dataset with the translated images
print(translated_dataset)
```

## Parameters

-   `input_dataset` (Dataset): A HuggingFace `Dataset` containing the source images to be translated.
-   `target_languages` (list of str): A list of language codes to translate the documents into (e.g., `["es", "fr", "zh"]`).
-   `input_images` (list of str, optional): You can also provide a list of image file paths directly instead of a dataset.

## Output

The method returns a new HuggingFace `Dataset` containing the translated document images. Each row in the output dataset corresponds to a translated version of a source document.
