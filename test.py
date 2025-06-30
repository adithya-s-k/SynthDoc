from synthdoc import create_handwriting_samples, create_vqa_dataset

# All workflows now return Dataset objects
handwriting_result = create_handwriting_samples(text_content="Hello!")
vqa_result = create_vqa_dataset(documents=["Sample document"])

# Access as real HuggingFace Datasets
handwriting_dataset = handwriting_result.dataset  # datasets.Dataset
vqa_dataset = vqa_result.dataset  # datasets.Dataset

# Use all HuggingFace Dataset features
print(f"Schema: {handwriting_dataset.features}")
print(f"Length: {len(handwriting_dataset)}")
handwriting_dataset.save_to_disk("handwriting_data")

#push to huggingface
handwriting_dataset.push_to_hub("handwriting_data")
vqa_dataset.push_to_hub("vqa_data")