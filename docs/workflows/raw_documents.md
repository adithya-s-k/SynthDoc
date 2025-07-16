# Raw Document Generation

The raw document generation workflow allows you to create synthetic documents from scratch using a Large Language Model (LLM). This is useful for generating a large and diverse dataset for training document understanding models.

## Usage

To generate documents, use the `generate_raw_docs` method of the `SynthDoc` class.

```python
from synthdoc import SynthDoc

# Initialize SynthDoc (it will load from your .env file)
synth = SynthDoc()

# Generate 2 pages in English with a specific topic
dataset = synth.generate_raw_docs(
    language="en",
    num_pages=2,
    prompt="Generate a short article about the benefits of renewable energy."
)

# The output is a HuggingFace Dataset
print(dataset)
print(dataset[0])
```

## Parameters

-   `language` (str): The language for the document content (e.g., `"en"`, `"es"`, `"zh"`).
-   `num_pages` (int): The number of pages to generate.
-   `prompt` (str, optional): A specific prompt to guide the LLM's content generation. If not provided, a random topic will be used.

## Output

The method returns a HuggingFace `Dataset` object where each row represents a generated document page. The schema includes:

-   `image`: A PIL Image of the document page.
-   `image_path`: The path to the saved image file.
-   `page_number`: The page number.
-   `language`: The language of the content.
-   `prompt`: The prompt used for generation.
-   `content_preview`: A short preview of the text content.
-   Other metadata related to the document.
