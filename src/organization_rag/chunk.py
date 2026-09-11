from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


MARKDOWN_FILE = Path(
    "data/markdown/Placement Handbook for Session 2026-2027 (1).md"
)


def main() -> None:
    text = MARKDOWN_FILE.read_text(encoding="utf-8")

    document = Document(
        text=text,
        metadata={
            "source": MARKDOWN_FILE.name,
        },
    )

    splitter = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    nodes = splitter.get_nodes_from_documents([document])

    print(f"Created chunks: {len(nodes)}")
    print()

    for i, node in enumerate(nodes[:5], start=1):
        print(f"--- Chunk {i} ---")
        print(node.text[:500])
        print()


if __name__ == "__main__":
    main()