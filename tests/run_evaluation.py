import json
from pathlib import Path

from organization_rag.llamaindex_rag import LlamaIndexRAG


EVALUATION_FILE = Path(__file__).parent / "evaluation_questions.json"

REFUSAL_TEXT = (
    "The provided documents do not contain enough information to answer this."
)


def evaluate_case(rag: LlamaIndexRAG, case: dict) -> dict:
    answer, sources = rag.answer(
        question=case["question"],
        department=case["department"],
    )

    answer_lower = answer.lower()

    expected_behavior = case["expected_behavior"]

    if expected_behavior == "answer":
        keyword_hits = sum(
            1
            for keyword in case["expected_keywords"]
            if keyword.lower() in answer_lower
        )

        passed = keyword_hits == len(case["expected_keywords"])

    elif expected_behavior in {"refuse", "unauthorized"}:
        passed = REFUSAL_TEXT.lower() in answer_lower

        # Unauthorized access must also return no sources.
        if expected_behavior == "unauthorized":
            passed = passed and len(sources) == 0

    else:
        passed = False

    return {
        "id": case["id"],
        "question": case["question"],
        "department": case["department"],
        "expected_behavior": expected_behavior,
        "passed": passed,
        "sources": len(sources),
        "answer": answer,
    }


def main():
    with EVALUATION_FILE.open("r", encoding="utf-8") as file:
        cases = json.load(file)

    rag = LlamaIndexRAG()

    results = []

    print("=" * 80)
    print("ORGANIZATION RAG EVALUATION")
    print("=" * 80)

    for case in cases:
        print()
        print(f"Running test {case['id']}/{len(cases)}...")
        print(f"Question: {case['question']}")
        print(f"Department: {case['department']}")

        result = evaluate_case(rag, case)
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"Result: {status}")

    total = len(results)
    passed = sum(result["passed"] for result in results)

    answer_cases = [
        result
        for result in results
        if result["expected_behavior"] == "answer"
    ]

    answer_passed = sum(result["passed"] for result in answer_cases)

    security_cases = [
        result
        for result in results
        if result["expected_behavior"] == "unauthorized"
    ]

    security_passed = sum(result["passed"] for result in security_cases)

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    if total:
        print(f"Overall pass rate: {passed / total:.1%}")

    if answer_cases:
        print(
            f"Answer success rate: "
            f"{answer_passed / len(answer_cases):.1%}"
        )

    if security_cases:
        print(
            f"Authorization success rate: "
            f"{security_passed / len(security_cases):.1%}"
        )

    output_file = Path(__file__).parent / "evaluation_results.json"

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print()
    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()