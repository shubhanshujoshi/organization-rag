# Multi-format document ingestion

This update implements the document-engineering requirement:

- recursively discover documents under `data/raw/<department>/`
- parse PDF/DOCX with Docling
- parse XLSX with openpyxl
- parse XLS with xlrd
- parse CSV/TXT/MD with lightweight faithful readers
- convert all supported formats to Markdown
- retain department, source, document type and page/sheet metadata
- ingest multiple files into Qdrant
- expose a Streamlit drag-and-drop multi-file uploader

## Folder model

```text
data/raw/
├── placement/
├── hr/
├── finance/
└── academic/
```

## UI flow

```text
Upload multiple files
        ↓
Select department
        ↓
Save under data/raw/<department>/
        ↓
Parse
        ↓
Markdown
        ↓
Chunk
        ↓
Embedding
        ↓
Qdrant
        ↓
Department-filtered RAG
```

## Important behavior

The current prototype rebuilds the Qdrant collection after an upload batch. This keeps
the implementation deterministic for the academic demo. Later we can add incremental
upserts if required.
