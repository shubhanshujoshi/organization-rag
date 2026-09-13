from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from organization_rag.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_URL,
)


class LlamaIndexRAG:
    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL)

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
        )

        self.embed_model = OllamaEmbedding(
            model_name=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        self.llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=300.0,
        )

        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=3,
        )

    def answer(self, question: str) -> str:
        response = self.query_engine.query(question)
        return str(response)