
import csv
import time
import uuid

from pathlib import Path
from datetime import datetime, timezone

from retrieval import retrieve
from model_client import generate_stream
from evaluator import evaluate_answer


# ==========================================
# 1. CONFIGURATION
# ==========================================

DEFAULT_SETTING = "B"
MAX_CORRECTIONS = 1
MAX_COMPLETION_TOKENS = 1200

OUTPUT_FILE = Path("results/pipeline_results.csv")
ANSWERS_FILE = Path("results/pipeline_answers.txt")
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

CSV_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade"
]


# ==========================================
# 2. PROMPT CONSTRUCTION
# ==========================================

def build_prompt(question, passages):

    context = ""

    for i, passage in enumerate(
        passages,
        start=1
    ):

        context += f"""
PASSAGE {i}

Paper: {passage['paper_title']}
arXiv: {passage['arxiv_id']}
Version: {passage['version']}
Page: {passage['page']}

Content:
{passage['text']}

--------------------------------
"""

    return f"""
You are The Relay, a research paper assistant.

Answer the user's question using ONLY the supplied
research passages.

STRICT RULES:

1. Never use external knowledge or assumptions.
2. Never invent facts, citations or page numbers.
3. Every factual claim must be supported by the passages.
4. Cite factual claims using this exact format:

   [arXiv:1706.03762v7, page 4]

5. Use the correct arXiv ID, version and page.
6. If evidence is insufficient, explicitly state:

   "The available papers do not provide sufficient
   evidence to answer this question."

7. Distinguish between information explicitly stated
   in the papers and information not stated.
8. Keep the answer concise and technically accurate.
9. Do not return an empty answer.
10. Do not include internal reasoning.

AVAILABLE RESEARCH PASSAGES:

{context}

USER QUESTION:

{question}

Write the final answer now.
"""


# ==========================================
# 3. CORRECTION PROMPT
# ==========================================

def build_correction_prompt(
    question,
    old_answer,
    passages,
    evaluation
):

    original_prompt = build_prompt(
        question,
        passages
    )

    return f"""
{original_prompt}

YOUR PREVIOUS ANSWER FAILED VALIDATION.

Previous answer:
{old_answer}

Validation failure:
{evaluation.get('reason', 'Unknown validation error')}

Correct the answer.

Instructions:

1. Use only the supplied research passages.
2. Remove unsupported statements.
3. Correct invalid citations.
4. Every factual claim must have a valid citation.
5. If evidence is insufficient, refuse to answer.
6. Return a complete final answer.
7. Never return an empty response.
"""


# ==========================================
# 4. CSV LOGGING
# ==========================================

def save_result(row):

    file_exists = (
        OUTPUT_FILE.exists()
        and OUTPUT_FILE.stat().st_size > 0
    )

    with OUTPUT_FILE.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_COLUMNS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ==========================================
# 5. TOKEN ACCOUNTING
# ==========================================

def add_usage(current, additional):

    if additional is None:
        return current

    if current is None:
        return additional

    return current + additional


# ==========================================
# 6. MAIN PIPELINE
# ==========================================

def answer_question(
    question,
    setting=DEFAULT_SETTING
):

    request_id = str(uuid.uuid4())[:8]

    start = time.perf_counter()

    start_time = datetime.now(
        timezone.utc
    ).isoformat()

    retries = 0

    first_token_ms = None

    tokens_in = None
    tokens_out = None

    retrieval_ms = 0.0
    total_ms = 0.0

    status = "ok"
    answer_grade = ""

    answer = ""

    print("\n" + "=" * 60)
    print("THE RELAY - NEW REQUEST")
    print("=" * 60)

    print("Request ID:", request_id)
    print("Question:", question)
    print("Retrieval setting:", setting)

    try:

        # ==================================
        # STEP 1: RETRIEVAL
        # ==================================

        print(
            "\n[1/4] Retrieving research passages..."
        )

        retrieval_start = time.perf_counter()

        passages = retrieve(
            question,
            setting=setting
        )

        retrieval_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        print(
            "Passages retrieved:",
            len(passages)
        )

        print(
            "Retrieval time:",
            round(retrieval_ms, 2),
            "ms"
        )

        if not passages:
            raise RuntimeError(
                "No research passages retrieved"
            )

        # ==================================
        # STEP 2: ANSWER GENERATION
        # ==================================

        print(
            "\n[2/4] Generating answer using Groq..."
        )

        prompt = build_prompt(
            question,
            passages
        )

        response = generate_stream(
            prompt,
            max_tokens=MAX_COMPLETION_TOKENS
        )

        answer = response["text"].strip()

        if not answer:
            raise RuntimeError(
                "Groq returned an empty answer"
            )

        if response.get("first_token_ms") is not None:

            first_token_ms = (
                retrieval_ms
                + response["first_token_ms"]
            )

        tokens_in = response.get("tokens_in")
        tokens_out = response.get("tokens_out")

        print("Generation completed.")

        print(
            "Model:",
            response.get("model")
        )

        print(
            "Finish reason:",
            response.get("finish_reason")
        )

        # ==================================
        # STEP 3: ANSWER VALIDATION
        # ==================================

        print(
            "\n[3/4] Validating answer and citations..."
        )

        evaluation = evaluate_answer(
            answer,
            passages
        )

        print(
            "Validation:",
            evaluation.get("reason")
        )

        # ==================================
        # STEP 4: CORRECTION BRANCH
        # ==================================

        if not evaluation["valid"]:

            print(
                "\n[4/4] Answer failed validation."
            )

            if retries < MAX_CORRECTIONS:

                retries += 1

                print(
                    "Attempting one correction..."
                )

                correction_prompt = build_correction_prompt(
                    question,
                    answer,
                    passages,
                    evaluation
                )

                try:

                    corrected = generate_stream(
                        correction_prompt,
                        max_tokens=MAX_COMPLETION_TOKENS
                    )

                    corrected_answer = (
                        corrected["text"].strip()
                    )

                    if not corrected_answer:
                        raise RuntimeError(
                            "Correction returned empty content"
                        )

                    # Preserve original TTFT.
                    answer = corrected_answer

                    tokens_in = add_usage(
                        tokens_in,
                        corrected.get("tokens_in")
                    )

                    tokens_out = add_usage(
                        tokens_out,
                        corrected.get("tokens_out")
                    )

                    evaluation = evaluate_answer(
                        answer,
                        passages
                    )

                    print(
                        "Correction validation:",
                        evaluation.get("reason")
                    )

                except Exception as correction_error:

                    print(
                        "Correction error:",
                        str(correction_error)
                    )

                    evaluation = {
                        "valid": False,
                        "reason": "Correction generation failed",
                        "refused": False
                    }

            # Final refusal if validation still fails.

            if not evaluation["valid"]:

                answer = (
                    "The available papers do not provide "
                    "sufficient verified evidence to answer "
                    "this question."
                )

                answer_grade = "refused_correctly"

                print(
                    "Correction unsuccessful. Refusing."
                )

            elif evaluation.get("refused"):

                answer_grade = "refused_correctly"

            else:

                answer_grade = "answered"

        elif evaluation.get("refused"):

            answer_grade = "refused_correctly"

        else:

            answer_grade = "answered"

        # ==================================
        # FINAL TIMING
        # ==================================

        total_ms = (
            time.perf_counter() - start
        ) * 1000

    except Exception as exc:

        print(
            "\nPipeline error:",
            str(exc)
        )

        code = getattr(
            exc,
            "status_code",
            None
        )

        if code == 429:

            status = "rate_limited"

        elif "timeout" in str(exc).lower():

            status = "timeout"

        else:

            status = "error"

        answer_grade = "generation_error"

        answer = (
            "The request could not be completed. "
            "Please try again later."
        )

        total_ms = (
            time.perf_counter() - start
        ) * 1000

    # ======================================
    # 7. SAVE REQUEST
    # ======================================

    row = {

        "request_id": request_id,

        "scenario": "pipeline_baseline",

        "source": "real",

        "start_time": start_time,

        "first_token_ms": (
            round(first_token_ms, 2)
            if first_token_ms is not None
            else ""
        ),

        "total_ms": round(total_ms, 2),

        "status": status,

        "retries": retries,

        "tokens_in": (
            tokens_in
            if tokens_in is not None
            else ""
        ),

        "tokens_out": (
            tokens_out
            if tokens_out is not None
            else ""
        ),

        "retrieval_setting": setting,

        "answer_grade": answer_grade
    }

    save_result(row)

    # ======================================
    # 8. DISPLAY FINAL ANSWER
    # ======================================

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)
    with ANSWERS_FILE.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 70 + "\n")
        file.write(f"Request ID: {request_id}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Answer Grade: {answer_grade}\n")
        file.write(f"Retrieval Setting: {setting}\n")
        file.write("-" * 70 + "\n")
        file.write(answer + "\n")
        file.write("=" * 70 + "\n\n")

    print("\n" + "-" * 60)
    print("REQUEST METRICS")
    print("-" * 60)

    print("Status:", status)

    print(
        "First token:",
        row["first_token_ms"],
        "ms"
    )

    print(
        "Total time:",
        row["total_ms"],
        "ms"
    )

    print("Retries:", retries)

    print("Input tokens:", tokens_in)

    print("Output tokens:", tokens_out)

    print("Retrieval time:", round(retrieval_ms, 2), "ms")

    print("Retrieval setting:", setting)

    print("Answer grade:", answer_grade)

    print("\nResults saved to:", OUTPUT_FILE)

    return {
        "answer": answer,
        "status": status,
        "metrics": row
    }


# ==========================================
# 9. INTERACTIVE MODE
# ==========================================

if __name__ == "__main__":

    print("=" * 60)
    print("THE RELAY - RESEARCH ASSISTANT")
    print("=" * 60)

    print("Model: openai/gpt-oss-20b")
    print("Retrieval setting:", DEFAULT_SETTING)
    print("Type 'exit' to stop.")

    while True:

        question = input(
            "\nAsk a research question: "
        )

        if question.strip().lower() == "exit":
            break

        if not question.strip():

            print(
                "Please enter a question."
            )

            continue

        answer_question(question)

