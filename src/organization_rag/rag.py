from llama_index.llms.ollama import Ollama

from organization_rag.config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
)
from organization_rag.retrieve import retrieve


class OrganizationRAG:
    def __init__(self) -> None:
        self.llm = Ollama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            request_timeout=300.0,
        )

    def answer(self, question: str) -> str:
        results = retrieve(question)

        if not results:
            return (
                "The provided documents do not contain enough "
                "information to answer this."
            )

        context_parts = []

        for rank, result in enumerate(results, start=1):
            payload = result.payload

            context_parts.append(
                f"[Source {rank} | "
                f"Chunk {payload['chunk_id']} | "
                f"{payload['source']}]\n"
                f"{payload['text']}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""
You are an organization policy assistant.

Answer the user's question using ONLY the provided context.

STRICT GROUNDING RULES:
1. Do not use outside knowledge.
2. Do not invent or infer policies.
3. Every factual claim must be supported by the context.
4. Cite factual claims using [Source N].
5. If the context does not contain enough information, say:
   "The provided documents do not contain enough information to answer this."

Retrieved context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.complete(prompt)

        return str(response)