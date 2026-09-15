from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from organization_rag.config import CHUNK_SIZE, CHUNK_OVERLAP


MARKDOWN_FILE = Path(
    "data/markdown/Placement Handbook for Session 2026-2027 (1).md"
)


def chunk_documents(documents: list[Document]):
    """Split LlamaIndex documents using the project's standard chunk settings."""
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return splitter.get_nodes_from_documents(documents)


def main() -> None:
    text = MARKDOWN_FILE.read_text(encoding="utf-8")

    document = Document(
        text=text,
        metadata={
            "source": MARKDOWN_FILE.name,
        },
    )

    nodes = chunk_documents([document])

    print(f"Created chunks: {len(nodes)}")
    print()

    for i, node in enumerate(nodes[:5], start=1):
        print(f"--- Chunk {i} ---")
        print(node.text[:500])
        print()


if __name__ == "__main__":
    main()
