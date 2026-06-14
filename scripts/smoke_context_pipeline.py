"""Manual smoke test for context expansion and assembly.

This script exercises the retrieval expansion and context assembly flow
without relying on pytest. It uses a lightweight fake vector store and
prints the assembled context output.
"""

from src.context.expander import NeighborContextExpander
from src.context.ContextAssembler import SimpleContextChunkAssembler
from src.core.search import SearchResult


class MockStore:
    """Fake vector store for neighbor chunk retrieval."""

    def __init__(self, chunks_by_doc):
        self.chunks_by_doc = chunks_by_doc

    def get_chunks_by_range(self, document_id: str, start: int, end: int):
        return [
            c
            for c in self.chunks_by_doc.get(document_id, [])
            if c.chunk_index is not None and start <= c.chunk_index <= end
        ]


def main() -> None:
    """Run a small smoke pipeline showing expander + assembler behavior."""
    seed_chunk = SearchResult(
        chunk_id="doc1:1",
        score=0.9,
        text="This is the center chunk.",
        source_file="doc1",
        chunk_index=1,
        start_page=1,
        end_page=1,
        section_title="Overview",
    )
    left_neighbor = SearchResult(
        chunk_id="doc1:0",
        score=0.75,
        text="This is the left neighbor.",
        source_file="doc1",
        chunk_index=0,
        start_page=1,
        end_page=1,
        section_title="Overview",
    )
    right_neighbor = SearchResult(
        chunk_id="doc1:2",
        score=0.65,
        text="This is the right neighbor.",
        source_file="doc1",
        chunk_index=2,
        start_page=1,
        end_page=1,
        section_title="Overview",
    )

    store = MockStore({"doc1": [left_neighbor, seed_chunk, right_neighbor]})
    expander = NeighborContextExpander(store, window_size=1)
    expanded_chunks = expander.expand([seed_chunk])

    print("Expanded chunks:")
    for chunk in expanded_chunks:
        print(f" - {chunk.chunk_id} | type={chunk.type} | seed={chunk.seed_chunk} | score={chunk.score}")

    assembler = SimpleContextChunkAssembler(max_characters=1200)
    context = assembler.assembleChunks(expanded_chunks)

    print("\nAssembled context:\n")
    print(context.text)
    print("\nSources:")
    for source in context.sources:
        print(f" - {source.chunk_id}")


if __name__ == "__main__":
    main()
