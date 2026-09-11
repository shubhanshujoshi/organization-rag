from pathlib import Path
import uuid

import requests
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient, models


MARKDOWN_FILE = Path(
    "data/markdown/Placement Handbook for Session 2026-2027 (1).md"
)

OLLAMA_URL = "http://localhost:11434/api/embed"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "organization_documents"


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "nomic-embed-text",
            "input": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def main() -> None:
    # Read the parsed Markdown.
    text = MARKDOWN_FILE.read_text(encoding="utf-8")

    document = Document(
        text=text,
        metadata={
            "source": MARKDOWN_FILE.name,
        },
    )

    # Split the document into manageable chunks.
    splitter = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    nodes = splitter.get_nodes_from_documents([document])

    print(f"Created chunks: {len(nodes)}")

    # Connect to Qdrant.
    client = QdrantClient(url=QDRANT_URL)

    # Create the collection.
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        ),
    )

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
                    "chunk_id": index,
                },
            )
        )

    # Insert all chunks into Qdrant.
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    info = client.get_collection(COLLECTION_NAME)

    print()
    print("Ingestion completed successfully.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Points stored: {info.points_count}")


if __name__ == "__main__":
    main()