from src.context.expander import NeighborContextExpander
from src.context.ContextAssembler import SimpleContextChunkAssembler
from src.core.search import SearchResult


def test_smoke_expander_and_assembler():
    """Smoke test covering a retrieval seed, expansion, and assembly."""
    seed_chunk = SearchResult(
        chunk_id="doc1:1",
        score=0.8,
        text="center chunk",
        source_file="doc1",
        chunk_index=1,
        start_page=1,
        section_title="Overview",
    )

    neighbor_left = SearchResult(
        chunk_id="doc1:0",
        score=0.6,
        text="left neighbor",
        source_file="doc1",
        chunk_index=0,
        start_page=1,
        section_title="Overview",
    )
    neighbor_right = SearchResult(
        chunk_id="doc1:2",
        score=0.5,
        text="right neighbor",
        source_file="doc1",
        chunk_index=2,
        start_page=1,
        section_title="Overview",
    )

    class MockStore:
        def __init__(self, chunks_by_doc):
            self.chunks_by_doc = chunks_by_doc

        def get_chunks_by_range(self, document_id, start, end):
            return [
                c
                for c in self.chunks_by_doc.get(document_id, [])
                if c.chunk_index is not None and start <= c.chunk_index <= end
            ]

    store = MockStore({"doc1": [neighbor_left, seed_chunk, neighbor_right]})
    expander = NeighborContextExpander(store, window_size=1)

    expanded = expander.expand([seed_chunk])
    assert len(expanded) == 3
    assert expanded[0].type == "retrieved"
    assert expanded[1].type == "expanded"
    assert expanded[2].type == "expanded"

    assembler = SimpleContextChunkAssembler(max_characters=1000)
    context = assembler.assembleChunks(expanded)

    assert "Document: doc1" in context.text
    assert "Page: 1" in context.text
    assert "Text: left neighbor" in context.text
    assert "Text: center chunk" in context.text
    assert "Text: right neighbor" in context.text
    assert len(context.sources) == 3
