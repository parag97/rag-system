from src.context.expander import NeighborContextExpander
from src.core.search import SearchResult


def test_neighbor_expander_smoke():
    # One retrieved chunk in the middle; vector store returns neighbors 0..2
    center = SearchResult(
        chunk_id="doc1:1",
        score=0.9,
        text="center",
        source_file="doc1",
        chunk_index=1,
    )

    all_chunks = [
        SearchResult(chunk_id="doc1:0", score=0.5, text="left", source_file="doc1", chunk_index=0),
        center,
        SearchResult(chunk_id="doc1:2", score=0.6, text="right", source_file="doc1", chunk_index=2),
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
    expander = NeighborContextExpander(store, window_size=1)

    expanded = expander.expand([center])
    texts = [c.text for c in expanded]
    assert texts == ["left", "center", "right"]
    # scores should be unchanged by the expander (option A)
    scores = [c.score for c in expanded]
    assert scores == [0.5, 0.9, 0.6]
