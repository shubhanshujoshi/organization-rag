# Organization RAG — Requirements

## 1. Project Objective

Build a department-aware Retrieval-Augmented Generation (RAG)
application that allows authorized users to ask questions about
organizational documents.

The system should retrieve relevant information from departmental
documents and generate grounded answers using an LLM.

## 2. Input Documents

The system should support common organizational document formats:

- PDF
- DOCX
- XLSX / Excel

Original documents must remain unchanged.

Documents will be organized by department, for example:

- HR
- Finance
- Legal
- IT
- Sales

## 3. Document Processing

The ingestion pipeline must:

1. Discover source documents.
2. Parse documents while preserving structure and meaning.
3. Convert parsed content into faithful Markdown/text.
4. Preserve useful metadata such as:
   - department
   - source filename
   - document type
   - page number
   - section
5. Clean and normalize extracted content.
6. Split documents into retrieval-friendly chunks.
7. Generate embeddings for the chunks.
8. Store chunks, embeddings, and metadata in a vector database.

An open-source document parser such as Docling should be evaluated.

## 4. Retrieval

The application must:

1. Accept a natural-language question.
2. Determine the user's permitted department scope.
3. Apply access-control filtering before retrieved content reaches
   the language model.
4. Retrieve the most relevant chunks.
5. Provide retrieved context to the LLM.
6. Preserve source information so answers can include citations.

## 5. Generation

The system should use an Ollama-hosted model for answer generation.

Answers should:

- be grounded in retrieved documents
- avoid unsupported claims
- clearly indicate when information is unavailable
- provide source references where possible

## 6. Vector Database Comparison

The project should evaluate multiple vector-storage approaches:

- Qdrant
- PostgreSQL with pgvector
- Chroma
- Supabase

The comparison should consider:

- setup complexity
- metadata filtering
- retrieval quality
- performance
- scalability
- developer experience
- suitability for departmental access control

One database will be selected as the primary implementation.

## 7. Application

The final system should provide a usable interface for:

- selecting/representing the user's department or access scope
- entering questions
- displaying answers
- displaying supporting sources
- handling questions for which sufficient information cannot be found

## 8. Testing and Evaluation

The project should evaluate:

### Document processing

- extraction accuracy
- table preservation
- heading/section preservation
- page/source traceability

### Retrieval

- Recall@K
- Precision@K
- Hit Rate
- MRR

### Generation

- correctness
- relevance
- faithfulness
- citation accuracy
- hallucination rate

### Security

Test that users cannot retrieve documents outside their
authorized department scope.

## 9. Engineering Requirements

The project should include:

- Python
- Git
- automated tests
- configuration through environment variables
- reproducible setup instructions
- clear project documentation
- modular source code

Secrets such as API keys must never be committed to Git.

## 10. Deliverables

The project should produce:

1. Working source code
2. Document ingestion pipeline
3. RAG application
4. Vector database comparison
5. Evaluation results
6. Automated tests
7. Architecture documentation
8. Final project report
9. Setup instructions