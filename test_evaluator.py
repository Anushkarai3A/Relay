
from src.evaluator import evaluate_answer

passages = [
    {
        "arxiv_id": "1706.03762",
        "version": 7,
        "page": 4
    },
    {
        "arxiv_id": "2410.00037",
        "version": 1,
        "page": 1
    },
    {
        "arxiv_id": "2507.07955",
        "version": 2,
        "page": 12
    }
]

test_cases = [
    "[arXiv:1706.03762v7, page 4]",
    "[arXiv:2410.00037v1, page 1]",
    "[arXiv:2507.07955v2, page 12]"
]

for citation in test_cases:
    result = evaluate_answer(
        f"Example answer. {citation}",
        passages
    )

    print(citation)
    print(result)
    print("-" * 50)
