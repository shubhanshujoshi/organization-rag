from organization_rag.llamaindex_rag import LlamaIndexRAG


def main():
    rag = LlamaIndexRAG()

    question = "How many placement attempts are allowed?"

    print(f"Question: {question}")
    print()
    print("Answer:")
    print("=" * 80)

    answer = rag.answer(question)

    print(answer)


if __name__ == "__main__":
    main()