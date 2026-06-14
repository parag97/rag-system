from src.context.expander import NeighborContextExpander
from src.context.ContextAssembler import SimpleContextChunkAssembler
from src.core.search import SearchResult


def test_integration_expander_assembler():
    # Setup vector store with three neighboring chunks
    all_chunks = [
        SearchResult(chunk_id="doc1:0", score=0.1, text="left-text", source_file="doc1", chunk_index=0, start_page=1),
        SearchResult(chunk_id="doc1:1", score=0.9, text="center-text", source_file="doc1", chunk_index=1, start_page=1),
        SearchResult(chunk_id="doc1:2", score=0.2, text="right-text", source_file="doc1", chunk_index=2, start_page=1),
    ]

    class MockStore:
        def __init__(self, chunks_by_doc):
            self.chunks_by_doc = chunks_by_doc

        def get_chunks_by_range(self, document_id, start, end):
            return [
                c
                for c in self.chunks_by_doc.get(document_id, [])
                if c.chunk_index is not None and start <= c.chunk_index <= end
            ]

    store = MockStore({"doc1": all_chunks})

    # initial retrieval returns only the center chunk
    center = all_chunks[1]
    expander = NeighborContextExpander(store, window_size=1)
    expanded = expander.expand([center])
    assert [c.type for c in expanded] == ["retrieved", "expanded", "expanded"]
    assert [c.seed_chunk for c in expanded] == ["doc1:1", "doc1:1", "doc1:1"]

    expanded_scores = {c.chunk_id: c.score for c in expanded}
    assert expanded_scores == {"doc1:0": 0.1, "doc1:1": 0.9, "doc1:2": 0.2}

    assembler = SimpleContextChunkAssembler(max_characters=1000)
    ctx = assembler.assembleChunks(expanded)

    assert "left-text" in ctx.text
    assert "center-text" in ctx.text
    assert "right-text" in ctx.text
    assert "Document: doc1" in ctx.text
