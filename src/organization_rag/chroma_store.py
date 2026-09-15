from pathlib import Path

import chromadb

from organization_rag.config import TOP_K
from organization_rag.embeddings import get_embedding


CHROMA_DIR = Path("data/chroma")
COLLECTION_NAME = "organization_documents"


class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def reset(self):
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add_chunks(self, chunks):
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        for i, chunk in enumerate(chunks):
            text = chunk.text
            metadata = dict(chunk.metadata)

            ids.append(f"chunk-{i}")
            documents.append(text)

            # Chroma metadata values need to be simple scalar types.
            clean_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    clean_metadata[key] = str(value)

            metadatas.append(clean_metadata)
            embeddings.append(get_embedding(text))

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(ids)

    def search(self, query, department=None, top_k=TOP_K):
        query_embedding = get_embedding(query)

        where = None
        if department:
            where = {
                "department": department
            }

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return result
