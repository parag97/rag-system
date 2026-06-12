# Current Status

The repository currently implements the dense-retrieval foundation:

- PDF loading into validated document/page models
- Recursive document chunking
- Separate document and query embedding boundaries
- Deterministic document and Qdrant point identities
- Qdrant collection validation, vector upsert, and dense search
- End-to-end ingestion orchestration with stage metrics

The hybrid retrieval, reranking, generation, API, and observability components
shown below are the intended target architecture, not completed components.

## Ingestion Identity And Lifecycle

Document IDs are derived from source content, and chunk point IDs are
deterministic UUIDs derived from document ID plus chunk index. Re-ingesting the
same source with the same chunking strategy therefore replaces matching Qdrant
points.

This does not yet provide complete document replacement semantics. If source
content changes, or a chunking change produces fewer chunks, older points remain
until explicitly deleted. A document lifecycle service and ingestion manifest
are planned to support atomic replace, delete, versioning, and stale-point
cleanup.

# Target Architecture

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
├── scripts/
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

# Delivery Roadmap

1. **Harden dense retrieval**
   - Add document replace/delete operations and ingestion manifests.
   - Add metadata filters, score thresholds, batching, and retrieval evaluation.
   - Separate unit, integration, and manual smoke-test execution.

2. **Add hybrid retrieval**
   - Define a common retriever interface for dense and sparse implementations.
   - Implement BM25 indexing/search and reciprocal-rank or weighted fusion.
   - Measure recall and ranking quality against an evaluation dataset.

3. **Add reranking and context construction**
   - Introduce candidate and ranked-result models.
   - Add a BGE reranker behind an interface.
   - Build token-budgeted context with source/page citations.

4. **Add generation and API**
   - Add an LLM inference interface and vLLM adapter.
   - Expose ingestion, retrieval, and answer endpoints through FastAPI.
   - Add request validation, background ingestion jobs, and failure handling.

5. **Add production operations**
   - Instrument pipeline stages with OpenTelemetry and Prometheus.
   - Add Langfuse traces, dashboards, deployment configuration, and load tests.
