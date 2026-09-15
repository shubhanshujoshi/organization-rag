from __future__ import annotations

from pathlib import Path

from organization_rag.document_loader import discover_documents
from organization_rag.ingest_all import ingest_documents


RAW_DIR = Path("data/raw")

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".csv",
    ".txt",
    ".md",
}


def save_uploaded_files(
    uploaded_files,
    department: str,
) -> list[Path]:
    """Save Streamlit UploadedFile objects into the department folder."""
    department_dir = RAW_DIR / department
    department_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix}")

        destination = department_dir / Path(uploaded_file.name).name
        destination.write_bytes(uploaded_file.getbuffer())
        saved_files.append(destination)

    return saved_files


def ingest_uploaded_files(
    uploaded_files,
    department: str,
    progress_callback=None,
) -> dict:
    """
    Save uploads and rebuild the index from ALL documents under data/raw/.

    This is important: uploading a new department must not remove documents
    that were uploaded previously for another department.
    """
    saved_files = save_uploaded_files(
        uploaded_files=uploaded_files,
        department=department,
    )

    all_source_documents = discover_documents(RAW_DIR)

    return ingest_documents(
        source_documents=all_source_documents,
        reset_collection=True,
        progress_callback=progress_callback,
    )
