# Architecture

```text
                           ┌──────────────┐
                           │  Documents   │
                           └──────┬───────┘
                                  │
                                  ▼

                           ┌──────────────┐
                           │  Chunking    │
                           └──────┬───────┘
                                  │
                                  ▼

                           ┌──────────────┐
                           │ Embeddings   │
                           └──────┬───────┘
                                  │
                                  ▼

                           ┌──────────────┐
                           │   Qdrant     │
                           │ Vector Store │
                           └──────┬───────┘
                                  │

────────────────────────────────────────────────────────────

User Query
     │
     ▼

┌──────────────┐
│ Query Rewrite│
└──────┬───────┘
       ▼

 ┌─────────────┐
 │ BM25 Search │
 └─────────────┘
        │

 ┌──────────────┐
 │ Vector Search│
 └──────────────┘

        ▼
 Candidate Pool
        ▼

┌──────────────┐
│   Reranker   │
└──────┬───────┘
       ▼

┌──────────────┐
│Context Builder│
└──────┬────────┘
       ▼

┌──────────────┐
│    vLLM      │
└──────┬───────┘
       ▼

Answer + Citations
```

# Tech Stack

## Retrieval

- Qdrant
- Sentence Transformers
- BM25

## Reranking

- BGE Reranker

## LLM Inference

- vLLM

## API Layer

- FastAPI

## Observability

- OpenTelemetry
- Langfuse
- Prometheus
- Grafana

## Containerization

- Docker
- Docker Compose

# Repository Structure

```text
rag-system/

├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── ingestion/
│   ├── chunking/
│   ├── embeddings/
│   ├── vectordb/
│   ├── retrieval/
│   ├── reranking/
│   ├── llm/
│   ├── api/
│   ├── observability/
│   └── evaluation/
│
├── tests/
├── configs/
├── docker/
├── notebooks/
├── docs/
└── README.md
```

# Build Order

```text
1. Qdrant
2. Embeddings
3. Document ingestion
4. Chunking
5. Retrieval
6. Evaluation
7. Hybrid Search
8. Reranker
9. FastAPI
10. vLLM
11. Observability
```
