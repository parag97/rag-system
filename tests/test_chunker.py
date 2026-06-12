# tests/chunking/test_recursive_chunker.py

from pathlib import Path

from src.chunking.recursive_chunker import RecursiveChunker
from src.ingestion.loaders.pdf_loader import PDFLoader


CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
PDF_PATH = Path("data/raw/amazon_annual_report.pdf")


def get_chunks():
    """Helper to load the PDF and create chunks."""
    loader = PDFLoader()
    chunker = RecursiveChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    document = loader.load(PDF_PATH)
    chunks = chunker.create_chunks(document)

    return document, chunks


def test_pdf_exists():
    assert PDF_PATH.exists()


def test_document_loads_successfully():
    document, _ = get_chunks()

    assert document is not None
    assert len(document.pages) > 0


def test_chunks_are_created():
    _, chunks = get_chunks()

    assert len(chunks) > 0


def test_chunk_indices_are_sequential():
    _, chunks = get_chunks()

    for expected_index, chunk in enumerate(chunks):
        assert chunk.chunk_index == expected_index


def test_chunk_text_is_not_empty():
    _, chunks = get_chunks()

    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_chunk_size_limit_is_respected():
    _, chunks = get_chunks()

    for chunk in chunks:
        assert len(chunk.text) <= CHUNK_SIZE


def test_document_metadata_is_preserved():
    document, chunks = get_chunks()

    for chunk in chunks:
        assert chunk.document_id == document.document_id
        assert chunk.source_file == document.source_file


def test_page_numbers_are_valid():
    document, chunks = get_chunks()

    valid_pages = {page.page_number for page in document.pages}

    for chunk in chunks:
        assert chunk.start_page in valid_pages
        assert chunk.end_page in valid_pages
        assert chunk.start_page <= chunk.end_page


def test_every_non_empty_page_has_a_chunk():
    document, chunks = get_chunks()

    pages_with_chunks = {chunk.start_page for chunk in chunks}

    for page in document.pages:
        if page.text.strip():
            assert page.page_number in pages_with_chunks


def test_chunking_is_deterministic():
    _, chunks_first = get_chunks()
    _, chunks_second = get_chunks()

    assert len(chunks_first) == len(chunks_second)

    for c1, c2 in zip(chunks_first, chunks_second):
        assert c1.chunk_index == c2.chunk_index
        assert c1.text == c2.text
        assert c1.start_page == c2.start_page
        assert c1.end_page == c2.end_page


def test_first_chunk_looks_reasonable():
    _, chunks = get_chunks()

    first_chunk = chunks[0]

    assert len(first_chunk.text) > 0
    assert first_chunk.chunk_index == 0
    assert first_chunk.start_page >= 1


def test_last_chunk_looks_reasonable():
    _, chunks = get_chunks()

    last_chunk = chunks[-1]

    assert len(last_chunk.text) > 0
    assert last_chunk.chunk_index == len(chunks) - 1
    assert last_chunk.start_page >= 1