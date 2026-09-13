import uuid
from pathlib import Path

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
from organization_rag.parser import parse_document


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
    """
    Infer department from the document's parent folder
    or filename.

    Examples:
        data/raw/placement/policy.pdf -> placement
        data/raw/hr/policy.docx       -> hr
        placement_handbook.pdf        -> placement
    """

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
    """
    Determine a simple document type from the filename.
    """

    filename = file_path.stem.lower()

    if "handbook" in filename:
        return "handbook"

    if "policy" in filename:
        return "policy"

    if "rule" in filename or "regulation" in filename:
        return "rules"

    return file_path.suffix.lower().replace(".", "") or "unknown"


def main() -> None:
    # ---------------------------------------------------------
    # 1. Discover source documents
    # ---------------------------------------------------------
    source_documents = discover_documents(RAW_DIR)

    if not source_documents:
        print("No supported documents found.")
        return

    print(f"Found {len(source_documents)} source document(s).")
    print()

    # ---------------------------------------------------------
    # 2. Parse every source document into Markdown
    # ---------------------------------------------------------
    markdown_files = []

    for source_file in source_documents:
        try:
            markdown_file = parse_document(
                input_file=source_file,
                output_dir=MARKDOWN_DIR,
            )

            markdown_files.append(
                (source_file, markdown_file)
            )

        except Exception as exc:
            print(
                f"Failed to parse {source_file.name}: {exc}"
            )

        print()

    if not markdown_files:
        print("No documents were successfully parsed.")
        return

    # ---------------------------------------------------------
    # 3. Create LlamaIndex documents
    # ---------------------------------------------------------
    documents = []

    for source_file, markdown_file in markdown_files:
        text = markdown_file.read_text(
            encoding="utf-8"
        )

        department = infer_department(source_file)
        document_type = infer_document_type(source_file)

        document = Document(
            text=text,
            metadata={
                "source": source_file.name,
                "document_type": document_type,
                "department": department,
            },
        )

        documents.append(document)

        print(
            f"Loaded: {source_file.name} "
            f"| department={department} "
            f"| type={document_type}"
        )

    # ---------------------------------------------------------
    # 4. Chunk all documents
    # ---------------------------------------------------------
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    print()
    print(f"Total chunks created: {len(nodes)}")

    # ---------------------------------------------------------
    # 5. Connect to Qdrant
    # ---------------------------------------------------------
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        print(
            f"Deleting existing collection: "
            f"{COLLECTION_NAME}"
        )
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=models.Distance.COSINE,
        ),
    )

    # ---------------------------------------------------------
    # 6. Generate embeddings
    # ---------------------------------------------------------
    points = []

    for index, node in enumerate(nodes):
        print(
            f"Embedding chunk "
            f"{index + 1}/{len(nodes)}..."
        )

        embedding = get_embedding(node.text)

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": node.text,
                    "source": node.metadata["source"],
                    "document_type": node.metadata[
                        "document_type"
                    ],
                    "department": node.metadata[
                        "department"
                    ],
                    "chunk_id": index,
                },
            )
        )

    # ---------------------------------------------------------
    # 7. Store vectors
    # ---------------------------------------------------------
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    # ---------------------------------------------------------
    # 8. Verify ingestion
    # ---------------------------------------------------------
    info = client.get_collection(COLLECTION_NAME)

    print()
    print("=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(nodes)}")
    print(f"Points stored: {info.points_count}")
    print()


if __name__ == "__main__":
    main()