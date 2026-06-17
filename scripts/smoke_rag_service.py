from src.core.config import load_config
from src.generator.lcel_generator import LCELGenerator
from src.embeddings.sentence_transformer_service import SentenceTransformerEmbeddingService
from src.embeddings.bm25_service import BM25EmbeddingService
from src.retrieval.retriever import Retriever
from src.vectordb.qdrant_store import QdrantVectorStore
from src.reranking.CrossEncoderReRanker import CrossEncoderReRanker
from src.context.expander import NeighborContextExpander
from src.context.ContextAssembler import SimpleContextChunkAssembler
from src.context.BasicContextBuilder import DefaultContextBuilder
from src.llm.google_studio import GoogleStudioGenerator
from src.generator.lcel_generator import LCELGenerator
from src.rag.RAGService import RAGService

import sys


def main() -> None:
	config = load_config()

	# Embedding and retrieval stack
	embedder = SentenceTransformerEmbeddingService(
		model_name=config.embedding.vector_model_name,
	)
	sparse = BM25EmbeddingService(
		model_name=config.embedding.sparse_model_name,
	)

	reranker = CrossEncoderReRanker(
		model_name=config.reranker.cross_encoder_model_name,
		top_k=config.reranker.top_k,
	)

	vector_store = QdrantVectorStore(
		host=config.qdrant.host,
		port=config.qdrant.port,
		top_k=config.qdrant.top_k,
		collection_name=config.qdrant.collection_name,
	)

	retriever = Retriever(
		dense_embedding_service=embedder,
		sparse_embedding_service=sparse,
		vector_store=vector_store,
		re_ranker=reranker,
	)

	# Context builder: expand + assemble
	expander = NeighborContextExpander(vector_store, window_size=2)
	assembler = SimpleContextChunkAssembler(max_characters=1500)
	context_builder = DefaultContextBuilder(expander, assembler)

	# Generator (LLM)
	llm = GoogleStudioGenerator().get_llm()
	generator = LCELGenerator(llm)

	rag = RAGService(
		retriever=retriever,
		context_builder=context_builder,
		generator=generator,
	)

	query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is Python?"

	print(f"Query: {query}")
	try:
		answer = rag.answer(query)
	except Exception as exc:
		print("Error during RAG answer:", exc)
		raise

	print("\nAnswer:\n")
	print(answer)


if __name__ == "__main__":
	main()
