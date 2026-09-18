# Organization RAG Assistant

A department-aware Retrieval-Augmented Generation (RAG) system for querying organizational documents using LlamaIndex, Ollama, vector databases, and Streamlit.

## 1. Project Overview

The Organization RAG Assistant allows users to upload organizational documents, process them into structured Markdown, create searchable vector representations, and ask natural-language questions.

The system is designed around departmental access control. Documents are associated with departments such as Placement, HR, Finance, Academic, Admissions, Administration, Library, Hostel, IT, and General.

When a user selects a department, retrieval is filtered to that department before the retrieved context is passed to the language model.

### Key objectives

- Support multiple document formats
- Preserve document structure during parsing
- Convert documents into Markdown
- Create semantic chunks for retrieval
- Generate embeddings using Ollama
- Compare multiple vector databases
- Implement department-level authorization
- Generate grounded answers using an LLM
- Provide source/page information for retrieved content
- Evaluate retrieval, generation, and authorization behavior

---

## 2. Architecture

```text
                    Organizational Documents
                 PDF / DOCX / XLSX / XLS /
                    CSV / TXT / Markdown
                              |
                              v
                    Document Discovery
                              |
                              v
                    Document Parser
                       (Docling)
                              |
                              v
                    Markdown Conversion
                              |
                              v
                 Page/Sheet-level Documents
                              |
                              v
                       Chunking
                 SentenceSplitter
                              |
                              v
                    Ollama Embeddings
                   nomic-embed-text
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
           Qdrant          Chroma       PostgreSQL
                                           + pgvector
              |               |               |
              +---------------+---------------+
                              |
                              v
                     Department Filter
                              |
                              v
                       Top-K Retrieval
                              |
                              v
                     LlamaIndex RAG
                              |
                              v
                    Ollama Cloud LLM
                    gpt-oss:20b-cloud
                              |
                              v
                    Grounded Response
                  + Source Information
                              |
                              v
                       Streamlit UI
