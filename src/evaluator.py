
import re


def evaluate_answer(answer, retrieved_passages):

    if not answer or not answer.strip():
        return {
            "valid": False,
            "reason": "Empty answer",
            "refused": False
        }

    # Detect an explicit refusal.
    refusal_phrases = [
        "the papers do not provide",
        "the available papers do not",
        "not stated in the papers",
        "cannot determine from the papers",
        "insufficient evidence"
    ]

    if any(
        phrase in answer.lower()
        for phrase in refusal_phrases
    ):
        return {
            "valid": True,
            "reason": "Evidence-based refusal",
            "refused": True
        }

    # Extract arXiv citations and page numbers.
    citation_pattern = (
        r"arXiv:\s*(\d{4}\.\d{5})"
        r"v(\d+)\s*,?\s*"
        r"p(?:age)?\s*(\d+)"
    )

    citations = re.findall(
        citation_pattern,
        answer,
        re.IGNORECASE
    )

    if not citations:
        return {
            "valid": False,
            "reason": "No valid paper citations",
            "refused": False
        }

    # Build the set of sources actually retrieved.
    valid_sources = {
        (
            str(p["arxiv_id"]),
            int(p["version"]),
            int(p["page"])
        )
        for p in retrieved_passages
    }

    # Verify every extracted citation.
    for citation in citations:

        source = (
            citation[0],
            int(citation[1]),
            int(citation[2])
        )

        if source not in valid_sources:
            return {
                "valid": False,
                "reason": f"Unsupported citation: {source}",
                "refused": False
            }

    return {
        "valid": True,
        "reason": "Citations match retrieved passages",
        "refused": False
    }
