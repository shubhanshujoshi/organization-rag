from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
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

    def answer(
        self,
        question: str,
        department: str,
    ) -> tuple[str, list[dict]]:

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="department",
                    value=department,
                    operator=FilterOperator.EQ,
                )
            ]
        )

        retriever = self.index.as_retriever(
            similarity_top_k=3,
            filters=filters,
        )

        results = retriever.retrieve(question)

        if not results:
            return (
                "The provided documents do not contain enough "
                "information to answer this.",
                [],
            )

        context_parts = []
        sources = []

        for rank, result in enumerate(results, start=1):
            node = result.node

            source = node.metadata.get(
                "source",
                "Unknown source",
            )

            sources.append(
                {
                    "rank": rank,
                    "source": source,
                    "score": result.score,
                    "department": node.metadata.get(
                        "department",
                        department,
                    ),
                }
            )

            context_parts.append(
                f"[Source {rank} | {source}]\n"
                f"{result.text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
You are an organization policy assistant.

Answer the user's question using ONLY the provided context.

STRICT GROUNDING RULES:
1. Do not use outside knowledge.
2. Do not invent policies or information.
3. Every factual claim must be supported by the context.
4. Cite factual claims using [Source N].
5. If the context does not contain enough information, say:
"The provided documents do not contain enough information to answer this."

Authorized department:
{department}

Retrieved context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.complete(prompt)

        return str(response), sources