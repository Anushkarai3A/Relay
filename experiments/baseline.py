
import os
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from groq import Groq

# ==========================================
# 1. CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(
    api_key=API_KEY,
    max_retries=0,
    timeout=30.0
)

OUTPUT = Path("results/results.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

SCENARIO = "steady_baseline"
SOURCE = "real"

# ==========================================
# 2. TEST QUESTIONS
# ==========================================

questions = [
    "Explain the attention mechanism in 2 sentences.",
    "What is multi-head attention? Answer briefly.",
    "Why are positional encodings needed? Answer briefly.",
    "Explain encoder-decoder architecture in 2 sentences.",
    "What is autoregressive generation? Answer briefly.",
    "Explain residual connections briefly.",
    "What is layer normalization? Answer briefly.",
    "Compare self-attention and cross-attention briefly.",
    "What is parallel sequence processing? Answer briefly.",
    "Explain training versus inference in 2 sentences."
]

# ==========================================
# 3. CSV CONFIGURATION
# ==========================================

columns = [
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

file_exists = OUTPUT.exists() and OUTPUT.stat().st_size > 0

# ==========================================
# 4. RUN BASELINE EXPERIMENT
# ==========================================

print("=" * 55)
print("THE RELAY - GROQ BASELINE EXPERIMENT")
print("=" * 55)

print("Model:", MODEL)
print("Requests:", len(questions))
print("Source:", SOURCE)
print()

with OUTPUT.open(
    "a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=columns
    )

    if not file_exists:
        writer.writeheader()

    for i, question in enumerate(questions, start=1):

        request_id = f"baseline-v2-{i}"

        start_time = datetime.now(
            timezone.utc
        ).isoformat()

        start = time.perf_counter()

        first_token = None
        tokens_in = None
        tokens_out = None
        finish_reason = None
        status = "ok"
        error_message = None

        print("-" * 55)
        print(f"Request {i}/{len(questions)}")
        print("Question:", question)

        # ==================================
        # 5. SEND STREAMING REQUEST
        # ==================================

        try:

            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                temperature=0,
                max_completion_tokens=1024,
                reasoning_effort="low",
                reasoning_format="hidden",
                stream=True
            )

            for chunk in stream:

                # Capture generated content and reasoning.
                if chunk.choices:

                    choice = chunk.choices[0]
                    delta = choice.delta

                    content = delta.content
                    if content and first_token is None:
                        first_token = time.perf_counter()
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    delta_data = delta.model_dump(
                        exclude_none=True
                    )

                    generated = any(
                        delta_data.get(key)
                        for key in [
                            "content",
                            "reasoning",
                            "reasoning_content",
                            "tool_calls",
                            "refusal"
                        ]
                    )

                    # Time to first generated event.
                    if generated and first_token is None:
                        first_token = time.perf_counter()

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                # Capture usage when the provider supplies it.
                groq_data = getattr(
                    chunk,
                    "x_groq",
                    None
                )

                usage = (
                    getattr(groq_data, "usage", None)
                    if groq_data else None
                )

                # Also support a top-level usage field.
                if usage is None:
                    usage = getattr(chunk, "usage", None)

                if usage:
                    tokens_in = usage.prompt_tokens
                    tokens_out = usage.completion_tokens

        # ==================================
        # 6. ERROR HANDLING
        # ==================================

        except Exception as exc:

            code = getattr(
                exc,
                "status_code",
                None
            )

            error_message = str(exc)

            if code == 429:
                status = "rate_limited"

            elif code in (408, 504) or "timeout" in str(exc).lower():
                status = "timeout"

            else:
                status = "error"

            print("Error:", error_message)

        # ==================================
        # 7. CALCULATE LATENCY
        # ==================================

        end = time.perf_counter()

        total_ms = round(
            (end - start) * 1000,
            1
        )

        if first_token is not None:
            first_token_ms = round(
                (first_token - start) * 1000,
                1
            )
        else:
            first_token_ms = ""

        # ==================================
        # 8. SAVE RAW RESULT
        # ==================================

        row = {
            "request_id": request_id,
            "scenario": SCENARIO,
            "source": SOURCE,
            "start_time": start_time,
            "first_token_ms": first_token_ms,
            "total_ms": total_ms,
            "status": status,
            "retries": 0,
            "tokens_in": (
                tokens_in if tokens_in is not None else ""
            ),
            "tokens_out": (
                tokens_out if tokens_out is not None else ""
            ),
            "retrieval_setting": "none",
            "answer_grade": ""
        }

        writer.writerow(row)

        # Save after every request.
        file.flush()

        # ==================================
        # 9. DISPLAY RESULTS
        # ==================================

        print("Status:", status)
        print("TTFT:", first_token_ms, "ms")
        print("Total time:", total_ms, "ms")
        print("Input tokens:", tokens_in)
        print("Output tokens:", tokens_out)
        print("Finish reason:", finish_reason)

        # Space requests to avoid intentional bursts.
        if i < len(questions):
            time.sleep(3)

# ==========================================
# 10. EXPERIMENT COMPLETE
# ==========================================

print()
print("=" * 55)
print("BASELINE EXPERIMENT COMPLETED")
print("=" * 55)

print("Total requests:", len(questions))
print("Results saved to:", OUTPUT)
print("Retries: Disabled")
print("Streaming: Enabled")
print("Model:", MODEL)

print()
print("Review results.csv before continuing.")
