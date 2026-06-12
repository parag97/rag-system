"""PDF loader implementation."""

from hashlib import sha256
from pathlib import Path

import fitz  # PyMuPDF

from src.ingestion.document import Document, Page
from src.ingestion.loaders.base import DocumentLoader


class PDFLoader(DocumentLoader):
    """Loads PDF files into the internal Document representation."""

    def load(self, pdf_path: str | Path) -> Document:
        """
        Load a PDF file and convert it into a Document.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A Document containing all extracted pages.
        """
        pdf_path = Path(pdf_path)
        document_id = self._document_id(pdf_path)

        with fitz.open(pdf_path) as pdf:
            pages = [
                Page(
                    page_number=page_number,
                    text=page.get_text().strip(),
                )
                for page_number, page in enumerate(pdf, start=1)
            ]

        return Document(
            document_id=document_id,
            source_file=str(pdf_path),
            pages=pages,
        )

    @staticmethod
    def _document_id(pdf_path: Path) -> str:
        """Build a deterministic, collision-resistant ID from file content."""
        digest = sha256()
        with pdf_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return f"{pdf_path.stem}-{digest.hexdigest()[:32]}"
