# Organization RAG — Architecture Decisions

## Decision 1 — Python 3.12

### Choice

Use Python 3.12 for the project.

### Reason

Python 3.12 provides a stable environment with broad compatibility
across the document-processing, RAG, embedding, and vector-database
libraries that will be evaluated.

---

## Decision 2 — Docling for Document Parsing

### Choice

Evaluate Docling as the primary open-source document parser.

### Reason

The project requires faithful conversion of organizational documents
into structured text/Markdown.

The parser should preserve:

- headings
- paragraphs
- lists
- tables
- page information
- document structure

Docling will therefore be evaluated for extraction quality across
PDF, DOCX, and other supported document formats.

Fallback parsers may be introduced where necessary.

---

## Decision 3 — LlamaIndex for RAG Orchestration

### Choice

Use LlamaIndex as the main RAG orchestration framework.

### Reason

The assignment specifically requires a LlamaIndex-based RAG
application.

LlamaIndex will be responsible for coordinating:

- document/node representation
- indexing
- retrieval
- context construction
- LLM interaction

Application-specific logic such as authorization and metadata
management will remain under our control rather than being delegated
entirely to the framework.

---

## Decision 4 — Qdrant as Initial Vector Database

### Choice

Use Qdrant as the initial vector database implementation.

### Reason

The project requires comparison of multiple vector-storage
approaches.

Qdrant is a suitable starting point because the application requires
metadata-aware retrieval and department-level filtering.

The choice is provisional.

The final vector database will be selected after comparing:

- retrieval quality
- metadata filtering
- performance
- setup complexity
- scalability
- developer experience

The alternatives are:

- PostgreSQL + pgvector
- Chroma
- Supabase

---

## Decision 5 — Department Metadata

### Choice

Attach department information to every document and chunk.

### Reason

Department metadata is required for access control.

Example:

```text
department = HR