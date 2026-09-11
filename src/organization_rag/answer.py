import requests

from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient


OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "organization_documents"


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": "nomic-embed-text",
            "input": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def main() -> None:
    question = input("Enter your question: ")

    # 1. Convert question into an embedding.
    query_embedding = get_embedding(question)

    # 2. Retrieve relevant chunks from Qdrant.
    client = QdrantClient(url=QDRANT_URL)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=3,
        with_payload=True,
    )

    # 3. Build context from retrieved chunks.
    context_parts = []

    for rank, result in enumerate(results.points, start=1):
        chunk_id = result.payload["chunk_id"]
        source = result.payload["source"]
        text = result.payload["text"]

        context_parts.append(
            f"[Source {rank} | Chunk {chunk_id} | {source}]\n{text}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Display retrieved sources so we can inspect retrieval quality.
    print("\nRetrieved sources:")
    print("=" * 80)

    for rank, result in enumerate(results.points, start=1):
        print(
            f"[Source {rank}] "
            f"Chunk {result.payload['chunk_id']} "
            f"(score: {result.score:.4f})"
        )

    # 4. Initialize the Ollama Cloud LLM.
    llm = Ollama(
        model="gpt-oss:20b-cloud",
        base_url="http://localhost:11434",
        request_timeout=300.0,
    )

    # 5. Build a strictly grounded prompt.
    prompt = f"""
You are an organization policy assistant.

Answer the user's question using ONLY the provided context.

STRICT GROUNDING RULES:
1. Do not use outside knowledge.
2. Do not invent or infer policies that are not explicitly supported.
3. Every factual claim must be supported by the provided sources.
4. Cite the source after each factual statement using [Source N].
5. If the retrieved context does not contain enough information, say:
   "The provided documents do not contain enough information to answer this."

Retrieved context:
{context}

Question:
{question}

Answer:
"""

    # 6. Generate the answer.
    response = llm.complete(prompt)

    print("\nAnswer:")
    print("=" * 80)
    print(response)


if __name__ == "__main__":
    main()