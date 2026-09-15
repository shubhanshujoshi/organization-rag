from qdrant_client import QdrantClient

from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

from organization_rag.access_control import (
    is_valid_department,
    normalize_department,
)
from organization_rag.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    TOP_K,
)


class LlamaIndexRAG:
    """Department-aware RAG using LlamaIndex, Qdrant and Ollama."""

    def __init__(self):
        # Connect to Qdrant
        self.client = QdrantClient(url=QDRANT_URL)

        # Configure Qdrant vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
        )

        # Configure embedding model
        self.embed_model = OllamaEmbedding(
            model_name=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # Configure Ollama Cloud LLM
        self.llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=120.0,
        )

        # Load existing vectors from Qdrant
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

    def answer(
        self,
        question: str,
        department: str,
    ) -> dict:
        """
        Answer a question using only documents belonging
        to the user's authorized department.
        """

        # ---------------------------------------------------------
        # 1. Normalize department
        # ---------------------------------------------------------
        department = normalize_department(department)

        # ---------------------------------------------------------
        # 2. Validate department
        # ---------------------------------------------------------
        if not is_valid_department(department):
            return {
                "answer": (
                    "I cannot answer this question because "
                    "the selected department is not authorized."
                ),
                "sources": [],
            }

        # ---------------------------------------------------------
        # 3. Create department metadata filter
        # ---------------------------------------------------------
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="department",
                    value=department,
                    operator=FilterOperator.EQ,
                )
            ]
        )

        # ---------------------------------------------------------
        # 4. Create retriever
        # ---------------------------------------------------------
        retriever = self.index.as_retriever(
            similarity_top_k=TOP_K,
            filters=filters,
        )

        # ---------------------------------------------------------
        # 5. Retrieve authorized context
        # ---------------------------------------------------------
        retrieved_nodes = retriever.retrieve(question)

        # ---------------------------------------------------------
        # 6. Refuse if no authorized information exists
        # ---------------------------------------------------------
        if not retrieved_nodes:
            return {
                "answer": (
                    "I could not find relevant information in the "
                    "authorized documents for this department."
                ),
                "sources": [],
            }

        # ---------------------------------------------------------
        # 7. Build context for the LLM
        # ---------------------------------------------------------
        context_parts = []

        for i, node_with_score in enumerate(
            retrieved_nodes,
            start=1,
        ):
            node = node_with_score.node

            context_parts.append(
                f"[Source {i}]\n"
                f"{node.get_content()}"
            )

        context = "\n\n".join(context_parts)

        # ---------------------------------------------------------
        # 8. Grounded RAG prompt
        # ---------------------------------------------------------
        prompt = f"""
You are an internal organization information assistant.

Answer the user's question using ONLY the information provided
in the context below.

Do not use outside knowledge.
Do not invent facts.
Do not infer information that is not supported by the context.

The user is authorized to access documents from the
"{department}" department.

If the context does not contain enough information to answer
the question, clearly say that the information is not available
in the authorized documents.

When making factual claims, cite the relevant source using
[Source 1], [Source 2], etc.

Context:

{context}

User question:

{question}

Answer:
"""

        # ---------------------------------------------------------
        # 9. Generate answer using Ollama Cloud
        # ---------------------------------------------------------
        response = self.llm.complete(prompt)

        answer_text = response.text.strip()

        # ---------------------------------------------------------
        # 10. Build source information
        # ---------------------------------------------------------
        sources = []

        for i, node_with_score in enumerate(
            retrieved_nodes,
            start=1,
        ):
            node = node_with_score.node
            metadata = node.metadata

            sources.append(
                {
                    "rank": i,
                    "source": metadata.get(
                        "source",
                        "Unknown",
                    ),
                    "page": metadata.get(
                        "page",
                        "Unknown",
                    ),
                    "section": metadata.get(
                        "section",
                        None,
                    ),
                    "score": (
                        float(node_with_score.score)
                        if node_with_score.score is not None
                        else None
                    ),
                    "department": metadata.get(
                        "department",
                        "Unknown",
                    ),
                }
            )

        # ---------------------------------------------------------
        # 11. Return answer + sources
        # ---------------------------------------------------------
        return {
            "answer": answer_text,
            "sources": sources,
        }