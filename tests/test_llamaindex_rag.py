from organization_rag.llamaindex_rag import LlamaIndexRAG


def main():
    rag = LlamaIndexRAG()

    question = "How many placement attempts are allowed?"

    print("=" * 80)
    print("TEST 1: AUTHORIZED DEPARTMENT")
    print("=" * 80)
    print("Department: placement")
    print(f"Question: {question}")
    print()

    answer, sources = rag.answer(
        question=question,
        department="placement",
    )

    print("Answer:")
    print(answer)

    print()
    print("Sources:")

    for source in sources:
        print(
            f"[{source['rank']}] "
            f"{source['source']} "
            f"| Page: {source.get('page')} "
            f"| Score: {source['score']:.3f}"
        )

    print()
    print("=" * 80)
    print("TEST 2: UNAUTHORIZED DEPARTMENT")
    print("=" * 80)
    print("Department: hr")
    print(f"Question: {question}")
    print()

    answer, sources = rag.answer(
        question=question,
        department="hr",
    )

    print("Answer:")
    print(answer)

    print()
    print("Sources returned:", len(sources))


if __name__ == "__main__":
    main()