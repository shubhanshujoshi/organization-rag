from __future__ import annotations

import uuid
from pathlib import Path
from typing import Callable

import requests
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient, models

from organization_rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MARKDOWN_DIR,
    OLLAMA_EMBED_URL,
    QDRANT_URL,
)
from organization_rag.document_loader import discover_documents
from organization_rag.parser import parse_document, parse_document_pages


RAW_DIR = Path("data/raw")


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def infer_department(file_path: Path) -> str:
    """Infer department from the parent folder or filename."""
    parent_name = file_path.parent.name.lower()
    known_departments = {
        "placement",
        "hr",
        "finance",
        "academic",
        "admissions",
        "administration",
        "library",
        "hostel",
        "it",
    }

    if parent_name in known_departments:
        return parent_name

    filename = file_path.stem.lower()
    for department in known_departments:
        if department in filename:
            return department

    return "general"


def infer_document_type(file_path: Path) -> str:
    """Determine a simple document type from the filename."""
    filename = file_path.stem.lower()

    if "handbook" in filename:
        return "handbook"
    if "policy" in filename:
        return "policy"
    if "rule" in filename or "regulation" in filename:
        return "rules"

    return file_path.suffix.lower().replace(".", "") or "unknown"


def _create_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=models.Distance.COSINE,
        ),
    )


def ingest_documents(
    source_documents: list[Path],
    reset_collection: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """
    Parse, chunk, embed and index multiple documents.

    The same function is used by both the CLI pipeline and Streamlit upload UI.
    """
    source_documents = sorted(Path(path) for path in source_documents)

    if not source_documents:
        raise ValueError("No supported documents were provided.")

    def progress(message: str) -> None:
        print(message)
        if progress_callback:
            progress_callback(message)

    markdown_files = []

    for source_file in source_documents:
        try:
            markdown_file = parse_document(
                input_file=source_file,
                output_dir=MARKDOWN_DIR,
            )
            markdown_files.append((source_file, markdown_file))
        except Exception as exc:
            progress(
                f"Failed to parse {source_file.name}: {exc}"
            )

    if not markdown_files:
        raise RuntimeError("No documents were successfully parsed.")

    documents: list[Document] = []

    for source_file, _ in markdown_files:
        department = infer_department(source_file)
        document_type = infer_document_type(source_file)
        pages = parse_document_pages(source_file)

        for page_data in pages:
            metadata = {
                "source": source_file.name,
                "source_path": str(source_file),
                "document_type": document_type,
                "department": department,
                "page": page_data["page"],
            }

            if page_data.get("section"):
                metadata["section"] = page_data["section"]

            documents.append(
                Document(
                    text=page_data["text"],
                    metadata=metadata,
                )
            )

        progress(
            f"Loaded: {source_file.name} | "
            f"department={department} | "
            f"type={document_type} | "
            f"logical pages={len(pages)}"
        )

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(documents)

    progress(f"Total chunks created: {len(nodes)}")

    client = QdrantClient(url=QDRANT_URL)

    if reset_collection:
        _create_collection(client)
    elif not client.collection_exists(COLLECTION_NAME):
        _create_collection(client)

    points = []

    for index, node in enumerate(nodes):
        progress(
            f"Embedding chunk {index + 1}/{len(nodes)}..."
        )

        embedding = get_embedding(node.text)

        payload = {
            "text": node.text,
            "source": node.metadata["source"],
            "source_path": node.metadata["source_path"],
            "document_type": node.metadata["document_type"],
            "department": node.metadata["department"],
            "page": node.metadata["page"],
            "chunk_id": index,
        }

        if "section" in node.metadata:
            payload["section"] = node.metadata["section"]

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload,
            )
        )

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    info = client.get_collection(COLLECTION_NAME)

    return {
        "documents": len(markdown_files),
        "logical_pages": len(documents),
        "chunks": len(nodes),
        "points": info.points_count,
    }


def main() -> None:
    source_documents = discover_documents(RAW_DIR)

    if not source_documents:
        print("No supported documents found.")
        return

    print(f"Found {len(source_documents)} source document(s).")
    print()

    stats = ingest_documents(source_documents)

    print()
    print("=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents: {stats['documents']}")
    print(f"Logical pages/sheets: {stats['logical_pages']}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Points stored: {stats['points']}")
    print()


if __name__ == "__main__":
    main()
