from pathlib import Path
import uuid

import requests
from qdrant_client import QdrantClient, models


MARKDOWN_FILE = Path(
    "data/markdown/Placement Handbook for Session 2026-2027 (1).md"
)

OLLAMA_URL = "http://localhost:11434/api/embed"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "organization_documents"


def main() -> None:
    text = MARKDOWN_FILE.read_text(encoding="utf-8")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "nomic-embed-text",
            "input": text[:5000],
        },
        timeout=120,
    )
    response.raise_for_status()

    embedding = response.json()["embeddings"][0]

    print(f"Embedding dimensions: {len(embedding)}")

    client = QdrantClient(url=QDRANT_URL)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": text[:5000],
                    "source": MARKDOWN_FILE.name,
                },
            )
        ],
    )

    info = client.get_collection(COLLECTION_NAME)

    print(f"Points in Qdrant: {info.points_count}")
    print("Direct Qdrant insertion successful.")


if __name__ == "__main__":
    main()