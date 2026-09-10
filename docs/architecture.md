# Organization RAG — System Architecture

## 1. High-Level Architecture

```text
                    ┌─────────────────────┐
                    │   Department Docs   │
                    │ PDF / DOCX / XLSX   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Document Scanner  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Document Parser      │
                    │ Docling / fallback   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Markdown Normalizer │
                    │ + Metadata          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Chunking            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Embedding Model     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Vector Database     │
                    │ Qdrant (primary)    │
                    └──────────┬──────────┘
                               │
                         Retrieval
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Access Control      │
                    │ Department Filter   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ LlamaIndex          │
                    │ RAG Orchestration   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Ollama LLM          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Answer + Citations  │
                    └─────────────────────┘