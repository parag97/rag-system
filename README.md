# RAG System

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI, Qdrant, and Google Gemini. Supports hybrid dense + sparse retrieval, cross-encoder reranking, neighbor context expansion, and automatic document sync via a filesystem watcher.

---

## Features

- **Hybrid search** — combines dense (Sentence Transformers) and sparse (BM25) embeddings with reciprocal rank fusion
- **Cross-encoder reranking** — refines retrieval results using a cross-encoder model
- **Context expansion** — enriches retrieved chunks with neighboring chunks from the same document
- **Automatic ingestion** — watchdog-based folder watcher syncs new, modified, and deleted PDFs on startup and at runtime
- **Document management API** — REST endpoints to ingest, list, inspect, and delete documents
- **Content-addressed identity** — document and chunk IDs are derived from file content, making re-ingestion idempotent
- **Config-driven** — all tunable parameters live in `configs/dev.yml`

---

## Architecture

```
PDF files (data/raw/)
        │
        ▼
  [ WatcherService ]  ──── filesystem events (watchdog)
        │
        ▼
  [ DocumentSyncService ]  ──── hash-based change detection
        │
        ▼
  [ DocumentIngestionService ]
        │
        ├── PDFLoader       (PyMuPDF)
        ├── RecursiveChunker
        ├── SentenceTransformerEmbeddingService  (dense)
        ├── BM25EmbeddingService                 (sparse)
        └── QdrantVectorStore.upsert()

────────────────────────────────────────────

HTTP POST /query
        │
        ▼
  [ Retriever ]
        ├── embed query (dense + sparse)
        ├── QdrantVectorStore.search()  ──── RRF fusion
        └── CrossEncoderReRanker

        │
        ▼
  [ DefaultContextBuilder ]
        ├── NeighborContextExpander
        └── SimpleContextChunkAssembler

        │
        ▼
  [ LCELGenerator ]  ──── Gemini via LangChain
        │
        ▼
     Answer (str)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Vector store | Qdrant |
| Dense embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Sparse embeddings | `Qdrant/bm25` via FastEmbed |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Google Gemini 2.5 Flash (via LangChain) |
| PDF parsing | PyMuPDF |
| File watching | Watchdog |
| Config | PyYAML + Pydantic |

---

## Project Structure

```
rag-system/
├── configs/
│   └── dev.yml               # All tunable parameters
├── data/
│   ├── raw/                  # Drop PDFs here for ingestion
│   ├── processed/
│   └── document_registry.json
├── src/
│   ├── api/                  # FastAPI routes, schemas, dependencies
│   ├── chunking/             # RecursiveChunker
│   ├── container/            # ApplicationContainer (DI wiring)
│   ├── context/              # Expander, assembler, context builder
│   ├── core/                 # Chunk, SearchResult, AppConfig models
│   ├── embeddings/           # Dense + sparse embedding services
│   ├── generator/            # LCELGenerator, prompt templates
│   ├── ingestion/            # Loader, ingestion service, registry, watcher
│   ├── llm/                  # LLM base + Google Studio adapter
│   ├── rag/                  # RAGService orchestrator
│   ├── reranking/            # CrossEncoderReRanker
│   ├── retrieval/            # Retriever
│   └── vectordb/             # QdrantVectorStore
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## Quick Start

### 1. Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for Qdrant)

### 2. Clone and install

```bash
git clone https://github.com/your-username/rag-system.git
cd rag-system
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your Google API key:

```env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL=gemini-2.5-flash
```

### 4. Start Qdrant

```bash
docker-compose up -d
```

Qdrant will be available at `http://localhost:6333`.

### 5. Add documents

Drop PDF files into `data/raw/`. They will be ingested automatically when the server starts.

### 6. Start the API

```bash
uv run uvicorn src.api.app:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Configuration

All parameters are in `configs/dev.yml`:

```yaml
embedding:
  vector_model_name: sentence-transformers/all-MiniLM-L6-v2
  sparse_model_name: Qdrant/bm25

qdrant:
  host: localhost
  port: 6333
  collection_name: documents
  top_k: 5

chunking:
  chunk_size: 500
  chunk_overlap: 100

reranker:
  cross_encoder_model_name: cross-encoder/ms-marco-MiniLM-L-6-v2
  top_k: 2

context_assembler:
  max_characters: 1500

expander:
  expansion_window_size: 2

ingestion:
  watch_folder: data/raw
  registry_path: data/document_registry.json
  file_ready_max_retries: 30
  file_ready_retry_delay_seconds: 1
```

---

## API Reference

### Query

**`POST /query`**

Ask a question against your ingested documents.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What were Amazon revenue in 2023?"}'
```

```json
{
  "answer": "Amazon's net sales were $574.8 billion in 2023..."
}
```

---

### Document Management

**`POST /documents/ingest`** — Ingest a file by path

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -H "Content-Type: application/json" \
  -d '{"file_path": "data/raw/amazon_annual_report.pdf"}'
```

```json
{
  "document_id": "amazon_annual_report-a1b2c3d4...",
  "source_file": "data/raw/amazon_annual_report.pdf",
  "page_count": 88,
  "chunk_count": 412,
  "vector_count": 412
}
```

**`GET /documents`** — List all ingested documents

```bash
curl http://localhost:8000/documents
```

```json
{
  "documents": [
    {
      "source_file": "data/raw/amazon_annual_report.pdf",
      "document_id": "amazon_annual_report-a1b2c3d4...",
      "file_hash": "e3b0c44298fc..."
    }
  ],
  "total": 1
}
```

**`GET /documents/{document_id}`** — Get a single document

```bash
curl http://localhost:8000/documents/amazon_annual_report-a1b2c3d4
```

**`DELETE /documents/{document_id}`** — Remove from index and registry

```bash
curl -X DELETE http://localhost:8000/documents/amazon_annual_report-a1b2c3d4
```

```json
{
  "document_id": "amazon_annual_report-a1b2c3d4...",
  "deleted": true
}
```

**`GET /health`** — Health check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## How Ingestion Works

1. On startup, `WatcherService` scans `data/raw/` and syncs all PDFs
2. A SHA-256 hash of each file is compared against `document_registry.json`
3. New files are ingested; unchanged files are skipped; modified files are re-indexed
4. The watcher then monitors the folder for changes in real time
5. Deleted files are automatically removed from Qdrant and the registry

---

## Running Tests

```bash
uv run pytest
```

Integration tests (require Qdrant running) are excluded by default. To include them:

```bash
uv run pytest -m integration
```
