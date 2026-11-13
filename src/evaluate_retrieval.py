
import csv
import json
import time
from pathlib import Path

from retrieval import SETTINGS, retrieve


# ==========================================
# CONFIGURATION
# ==========================================

QUESTIONS_FILE = Path("data/evaluation_questions.json")

RESULTS_DIR = Path("results")
EVIDENCE_DIR = RESULTS_DIR / "evidence_passages"

OUTPUT_FILE = RESULTS_DIR / "retrieval_evaluation.csv"
SUMMARY_FILE = RESULTS_DIR / "retrieval_summary.json"
REVIEW_FILE = RESULTS_DIR / "evidence_review.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD QUESTIONS
# ==========================================

def load_questions():

    with QUESTIONS_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if len(questions) != 15:
        raise ValueError(
            f"Expected 15 questions, found {len(questions)}"
        )

    return questions


# ==========================================
# PAPER ID NORMALIZATION
# ==========================================

def normalize_paper_id(paper_id, version=None):

    paper_id = str(paper_id).strip()
    version = str(version or "").strip()

    if version and not version.startswith("v"):
        version = "v" + version

    if version and not paper_id.endswith(version):
        paper_id += version

    return paper_id


# ==========================================
# FIND EXPECTED PAPER RANK
# ==========================================

def get_target_rank(results, expected_paper):

    for rank, result in enumerate(results, start=1):

        retrieved_paper = normalize_paper_id(
            result["arxiv_id"],
            result["version"]
        )

        if retrieved_paper == expected_paper:
            return rank

    return None


# ==========================================
# SAVE FULL RETRIEVED PASSAGES
# ==========================================

def save_evidence(question, results):

    question_id = question["id"]

    evidence_file = EVIDENCE_DIR / f"{question_id}.txt"

    with evidence_file.open("w", encoding="utf-8") as file:

        file.write("=" * 75 + "\n")
        file.write(f"QUESTION ID: {question_id}\n")
        file.write("=" * 75 + "\n\n")

        file.write(f"Question: {question['question']}\n")
        file.write(f"Expected paper: {question['paper']}\n")
        file.write(f"Answerable: {question['answerable']}\n\n")

        file.write("Expected evidence:\n")
        file.write(question["expected_evidence"] + "\n\n")

        file.write(f"Retrieved passages: {len(results)}\n\n")

        for rank, result in enumerate(results, start=1):

            paper_id = normalize_paper_id(
                result["arxiv_id"],
                result["version"]
            )

            file.write("-" * 75 + "\n")
            file.write(f"RANK: {rank}\n")
            file.write(f"PAPER: {paper_id}\n")
            file.write(f"PAGE: {result['page']}\n")

            file.write(
                f"SIMILARITY SCORE: "
                f"{result['similarity_score']:.4f}\n"
            )

            if "rerank_score" in result:
                file.write(
                    f"RERANK SCORE: "
                    f"{result['rerank_score']:.4f}\n"
                )

            file.write("\nPASSAGE TEXT:\n\n")
            file.write(result["text"])
            file.write("\n\n")

    return evidence_file


# ==========================================
# CREATE EVIDENCE REVIEW TEMPLATE
# ==========================================

def create_review_template(questions):

    fieldnames = [
        "question_id",
        "setting",
        "expected_evidence_present",
        "evidence_page",
        "review_notes"
    ]

    # Preserve an existing manual review.
    if REVIEW_FILE.exists():
        print("\nExisting evidence review preserved.")
        return

    with REVIEW_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for question in questions:

            writer.writerow({
                "question_id": question["id"],
                "setting": "B",
                "expected_evidence_present": "",
                "evidence_page": "",
                "review_notes": ""
            })

    print("Evidence review template created:", REVIEW_FILE)


# ==========================================
# RUN RETRIEVAL EVALUATION
# ==========================================

def evaluate():

    questions = load_questions()

    records = []

    print("=" * 65)
    print("THE RELAY - RETRIEVAL EVALUATION")
    print("=" * 65)

    print("Questions:", len(questions))
    print("Settings:", list(SETTINGS.keys()))
    print("Total tests:", len(questions) * len(SETTINGS))

    # --------------------------------------
    # WARM UP
    # --------------------------------------

    print("\nWarming up retrieval models...")

    for setting in SETTINGS:

        retrieve(
            "What is the main contribution of this research paper?",
            setting
        )

        print(f"Setting {setting} ready.")

    # --------------------------------------
    # EVALUATION
    # --------------------------------------

    print("\nStarting evaluation...")

    for setting in SETTINGS:

        print("\n" + "=" * 65)
        print("SETTING:", setting)
        print("=" * 65)

        for question in questions:

            question_id = question["id"]
            query = question["question"]

            expected_paper = normalize_paper_id(
                question["paper"]
            )

            start = time.perf_counter()

            results = retrieve(
                query,
                setting
            )

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            target_rank = get_target_rank(
                results,
                expected_paper
            )

            paper_hit = target_rank is not None

            reciprocal_rank = (
                1 / target_rank
                if target_rank is not None
                else 0
            )

            retrieved_sources = []

            for result in results:

                paper_id = normalize_paper_id(
                    result["arxiv_id"],
                    result["version"]
                )

                retrieved_sources.append(
                    f"{paper_id}:page_{result['page']}"
                )

            record = {
                "question_id": question_id,
                "question": query,
                "expected_paper": expected_paper,
                "answerable": question["answerable"],
                "setting": setting,
                "retrieval_time_ms": round(elapsed_ms, 2),
                "retrieved_count": len(results),
                "target_paper_hit": paper_hit,
                "target_rank": target_rank or "",
                "reciprocal_rank": round(
                    reciprocal_rank, 4
                ),
                "retrieved_sources": "; ".join(
                    retrieved_sources
                )
            }

            records.append(record)

            # Save complete passages for setting B.
            if setting == "B":

                evidence_file = save_evidence(
                    question,
                    results
                )

                print(
                    f"Evidence saved: {evidence_file}"
                )

            print(
                f"{question_id} | "
                f"{elapsed_ms:.2f} ms | "
                f"Paper hit: {paper_hit} | "
                f"Rank: {target_rank or 'Not found'}"
            )

    # --------------------------------------
    # SAVE RETRIEVAL CSV
    # --------------------------------------

    fieldnames = [
        "question_id",
        "question",
        "expected_paper",
        "answerable",
        "setting",
        "retrieval_time_ms",
        "retrieved_count",
        "target_paper_hit",
        "target_rank",
        "reciprocal_rank",
        "retrieved_sources"
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    # --------------------------------------
    # CALCULATE SUMMARY
    # --------------------------------------

    summary = {}

    for setting in SETTINGS:

        setting_records = [
            row for row in records
            if row["setting"] == setting
        ]

        answerable_records = [
            row for row in setting_records
            if row["answerable"]
        ]

        hits = sum(
            row["target_paper_hit"]
            for row in answerable_records
        )

        count = len(answerable_records)

        hit_rate = hits / count if count else 0

        mean_reciprocal_rank = (
            sum(
                row["reciprocal_rank"]
                for row in answerable_records
            ) / count
            if count else 0
        )

        mean_time = (
            sum(
                row["retrieval_time_ms"]
                for row in setting_records
            ) / len(setting_records)
            if setting_records else 0
        )

        summary[setting] = {
            "total_questions": len(setting_records),
            "answerable_questions": count,
            "unanswerable_questions": (
                len(setting_records) - count
            ),
            "target_paper_hits": hits,
            "target_paper_hit_rate": round(
                hit_rate, 4
            ),
            "mean_reciprocal_rank": round(
                mean_reciprocal_rank, 4
            ),
            "mean_retrieval_time_ms": round(
                mean_time, 2
            )
        }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------
    # CREATE REVIEW TEMPLATE
    # --------------------------------------

    create_review_template(questions)

    # --------------------------------------
    # PRINT SUMMARY
    # --------------------------------------

    print("\n" + "=" * 65)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 65)

    for setting, metrics in summary.items():

        print(f"\nSETTING {setting}")

        print(
            f"Target paper hits: "
            f"{metrics['target_paper_hits']}/"
            f"{metrics['answerable_questions']}"
        )

        print(
            f"Hit rate: "
            f"{metrics['target_paper_hit_rate'] * 100:.2f}%"
        )

        print(
            f"Mean reciprocal rank: "
            f"{metrics['mean_reciprocal_rank']}"
        )

        print(
            f"Mean retrieval time: "
            f"{metrics['mean_retrieval_time_ms']:.2f} ms"
        )

    print("\nEvaluation completed.")
    print("CSV:", OUTPUT_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Evidence passages:", EVIDENCE_DIR)
    print("Evidence review:", REVIEW_FILE)


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    evaluate()
