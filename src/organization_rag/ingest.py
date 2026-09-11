import uuid

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


MARKDOWN_FILE = MARKDOWN_DIR / "Placement Handbook for Session 2026-2027 (1).md"

DEPARTMENT = "placement"


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


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load the processed Markdown document
    # ---------------------------------------------------------
    text = MARKDOWN_FILE.read_text(encoding="utf-8")

    document = Document(
        text=text,
        metadata={
            "source": MARKDOWN_FILE.name,
            "document_type": "placement_handbook",
            "department": DEPARTMENT,
        },
    )

    # ---------------------------------------------------------
    # 2. Split document into chunks
    # ---------------------------------------------------------
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents([document])

    print(f"Created chunks: {len(nodes)}")

    # ---------------------------------------------------------
    # 3. Connect to Qdrant
    # ---------------------------------------------------------
    client = QdrantClient(url=QDRANT_URL)

    # Delete existing collection so we can rebuild the index
    # with the new metadata.
    if client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)

    # ---------------------------------------------------------
    # 4. Create fresh Qdrant collection
    # ---------------------------------------------------------
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSIONS,
            distance=models.Distance.COSINE,
        ),
    )

    # ---------------------------------------------------------
    # 5. Generate embeddings and prepare Qdrant points
    # ---------------------------------------------------------
    points = []

    for index, node in enumerate(nodes):
        print(f"Embedding chunk {index + 1}/{len(nodes)}...")

        embedding = get_embedding(node.text)

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": node.text,
                    "source": MARKDOWN_FILE.name,
                    "document_type": "placement_handbook",
                    "department": DEPARTMENT,
                    "chunk_id": index,
                },
            )
        )

    # ---------------------------------------------------------
    # 6. Insert vectors into Qdrant
    # ---------------------------------------------------------
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    # ---------------------------------------------------------
    # 7. Verify ingestion
    # ---------------------------------------------------------
    info = client.get_collection(COLLECTION_NAME)

    print()
    print("Ingestion completed successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Department: {DEPARTMENT}")
    print(f"Points stored: {info.points_count}")


if __name__ == "__main__":
    main()