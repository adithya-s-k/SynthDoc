from synthdoc.models import RawDocumentGenerationConfig
from synthdoc.workflows import RawDocumentGenerator
from synthdoc.languages import Language

# Basic configuration
config = RawDocumentGenerationConfig(
    language=Language.EN,
    num_pages=2,
    prompt="Write about machine learning"
)

# Generate documents
generator = RawDocumentGenerator(save_dir="my_documents")
result = generator.process(config)

print(f"Generated {result.num_samples} pages")
print(f"Output: {result.metadata['output_structure']['output_dir']}")