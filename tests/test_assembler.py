from src.context.ContextAssembler import SimpleContextChunkAssembler
from src.core.search import SearchResult


def test_assembler_smoke_ordering_and_duplicates():
    chunks = [
        SearchResult(
            chunk_id="doc1:1",
            score=0.5,
            text="A",
            source_file="doc1",
            chunk_index=1,
            start_page=1,
            section_title="S",
            type="retrieved",
        ),
        SearchResult(
            chunk_id="doc1:2",
            score=0.9,
            text="B",
            source_file="doc1",
            chunk_index=2,
            start_page=1,
            section_title="S",
            type="retrieved",
        ),
        SearchResult(
            chunk_id="doc2:0",
            score=0.7,
            text="C",
            source_file="doc2",
            chunk_index=0,
            start_page=2,
            section_title="T",
            type="retrieved",
        ),
        # duplicate chunk_id should be ignored by assembler
        SearchResult(
            chunk_id="doc1:2",
            score=0.9,
            text="B",
            source_file="doc1",
            chunk_index=2,
            start_page=1,
            section_title="S",
            type="retrieved",
        ),
    ]

    assembler = SimpleContextChunkAssembler(max_characters=1000)
    ctx = assembler.assembleChunks(chunks)

    assert "Document: doc1" in ctx.text
    assert "Page: 1" in ctx.text

    ids = [s.chunk_id for s in ctx.sources]
    assert ids.count("doc1:2") == 1
    # highest score first: doc1:2 (0.9)
    assert ids[0] == "doc1:2"


def test_assembler_smoke_single_chunk():
    chunk = SearchResult(
        chunk_id="doc1:0",
        score=0.8,
        text="single",
        source_file="doc1",
        chunk_index=0,
        start_page=1,
        section_title="Intro",
        type="retrieved",
    )

    assembler = SimpleContextChunkAssembler(max_characters=1000)
    ctx = assembler.assembleChunks([chunk])

    assert "Document: doc1" in ctx.text
    assert "Text: single" in ctx.text
    assert ctx.sources == [chunk]


def test_assembler_respects_max_length_and_context_headers():
    chunks = [
        SearchResult(
            chunk_id="doc1:0",
            score=0.8,
            text="first",
            source_file="doc1",
            chunk_index=0,
            start_page=1,
            section_title="Intro",
            type="retrieved",
        ),
        SearchResult(
            chunk_id="doc1:1",
            score=0.7,
            text="second",
            source_file="doc1",
            chunk_index=1,
            start_page=2,
            section_title="Details",
            type="retrieved",
        ),
    ]

    assembler = SimpleContextChunkAssembler(max_characters=300)
    ctx = assembler.assembleChunks(chunks)

    assert "Document: doc1" in ctx.text
    assert "Page: 1" in ctx.text or "Page: 2" in ctx.text
    assert len(ctx.text) <= 300
    assert all(s.type == "retrieved" for s in ctx.sources)
