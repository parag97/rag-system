from src.ingestion.document_registry import (
    DocumentRegistry,
    RegistryEntry,
)

registry = DocumentRegistry(
    "data/document_registry.json"
)

registry.set(
    "data/raw/amazon.pdf",
    RegistryEntry(
        document_id="amazon-123",
        file_hash="abc",
    ),
)

entry = registry.get(
    "data/raw/amazon.pdf"
)

print(entry)

registry.remove(
    "data/raw/amazon.pdf"
)