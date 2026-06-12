"""Tests for RecursiveChunker."""

import unittest

from src.chunking.recursive_chunker import RecursiveChunker
from src.ingestion.document import Document, Page


def _make_document(
    pages: list[tuple[int, str]],
    *,
    document_id: str = "doc-1",
    source_file: str = "sample.txt",
) -> Document:
    return Document(
        document_id=document_id,
        source_file=source_file,
        pages=[Page(page_number=number, text=text) for number, text in pages],
    )


class TestRecursiveChunker(unittest.TestCase):
    def test_short_text_produces_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        document = _make_document([(1, "Hello, world!")])

        chunks = chunker.create_chunks(document)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Hello, world!")
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[0].document_id, "doc-1")
        self.assertEqual(chunks[0].source_file, "sample.txt")
        self.assertEqual(chunks[0].start_page, 1)
        self.assertEqual(chunks[0].end_page, 1)

    def test_long_text_splits_into_multiple_chunks(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        text = "word " * 30
        document = _make_document([(1, text)])

        chunks = chunker.create_chunks(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 50 for chunk in chunks))
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[-1].chunk_index, len(chunks) - 1)
        self.assertEqual(
            "".join(chunk.text for chunk in chunks).replace(" ", ""),
            text.replace(" ", ""),
        )

    def test_chunk_indices_are_sequential_across_pages(self):
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0)
        document = _make_document(
            [
                (1, "page one " * 10),
                (2, "page two " * 10),
            ]
        )

        chunks = chunker.create_chunks(document)

        self.assertGreater(len(chunks), 2)
        self.assertEqual(
            [chunk.chunk_index for chunk in chunks],
            list(range(len(chunks))),
        )
        self.assertTrue(any(chunk.start_page == 1 for chunk in chunks))
        self.assertTrue(any(chunk.start_page == 2 for chunk in chunks))

    def test_page_metadata_is_set_per_chunk(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        document = _make_document(
            [
                (3, "First page."),
                (7, "Second page."),
            ]
        )

        chunks = chunker.create_chunks(document)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].start_page, 3)
        self.assertEqual(chunks[0].end_page, 3)
        self.assertEqual(chunks[1].start_page, 7)
        self.assertEqual(chunks[1].end_page, 7)

    def test_overlap_appears_in_adjacent_chunks(self):
        chunker = RecursiveChunker(chunk_size=40, chunk_overlap=10)
        text = "abcdefghij" * 8
        document = _make_document([(1, text)])

        chunks = chunker.create_chunks(document)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn(chunks[0].text[-10:], chunks[1].text)


if __name__ == "__main__":
    unittest.main()
