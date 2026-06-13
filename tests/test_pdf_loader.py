"""Tests for PDF loader.

Validates that the PDFLoader correctly extracts pages from PDF files
and generates deterministic document IDs based on file content.
"""

from pathlib import Path

from src.ingestion.loaders.pdf_loader import PDFLoader

PDF_PATH = "data/raw/amazon_annual_report.pdf"


def test_pdf_loader_loads_document():
    document = PDFLoader().load(PDF_PATH)

    assert document.document_id.startswith("amazon_annual_report-")
    assert Path(document.source_file) == Path(PDF_PATH)
    assert len(document.pages) > 0
    assert all(page.text is not None for page in document.pages)


def test_pdf_loader_document_id_is_deterministic():
    loader = PDFLoader()

    first = loader.load(PDF_PATH)
    second = loader.load(PDF_PATH)

    assert first.document_id == second.document_id
