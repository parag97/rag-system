"""PDF loader implementation."""

from hashlib import sha256
from pathlib import Path

import fitz  # PyMuPDF

from src.ingestion.document import Document, Page
from src.ingestion.loaders.base import DocumentLoader


class PDFLoader(DocumentLoader):
    """Loads PDF files into the internal Document representation.

    This loader uses PyMuPDF to extract textual content page-by-page and
    produces a deterministic document identifier derived from the file
    contents to allow idempotent ingestion runs.
    """

    def load(self, pdf_path: str | Path) -> Document:
        """Load a PDF file and convert it into a Document.

        The method reads the PDF with PyMuPDF, extracts text from each
        page, strips surrounding whitespace, and constructs `Page`
        objects in reading order.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A Document containing all extracted pages.
        """
        pdf_path = Path(pdf_path)
        document_id = self._document_id(pdf_path)

        # Open the PDF and extract page text deterministically.
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
        """Build a deterministic, collision-resistant ID from file content.

        The returned string is composed of the file stem and the first 32
        hex characters of a SHA-256 digest of the file contents.
        """
        digest = sha256()
        with pdf_path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return f"{pdf_path.stem}-{digest.hexdigest()[:32]}"
