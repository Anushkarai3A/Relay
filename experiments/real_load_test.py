
import argparse
import csv
import os
import time
import uuid
import threading

from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from groq import Groq, APIStatusError


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "results.csv"
HEADERS_FILE = RESULTS_DIR / "groq_response_headers.csv"

load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

client = Groq(
    api_key=API_KEY,
    max_retries=0,
    timeout=60.0
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

HEADER_COLUMNS = [
    "request_id",
    "scenario",
    "status_code",
    "retry_after",
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-reset-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens"
]

header_lock = threading.Lock()


# ==================================================
# CSV UTILITIES
# ==================================================

def append_csv(path, columns, records):

    if not records:
        return

    file_exists = (
        path.exists()
        and path.stat().st_size > 0
    )

    with path.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
            extrasaction="raise"
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(records)


def extract_headers(headers):

    if headers is None:
        headers = {}

    return {
        "retry_after": headers.get("retry-after", ""),

        "x-ratelimit-limit-requests":
            headers.get("x-ratelimit-limit-requests", ""),

        "x-ratelimit-remaining-requests":
            headers.get("x-ratelimit-remaining-requests", ""),

        "x-ratelimit-reset-requests":
            headers.get("x-ratelimit-reset-requests", ""),

        "x-ratelimit-limit-tokens":
            headers.get("x-ratelimit-limit-tokens", ""),

        "x-ratelimit-remaining-tokens":
            headers.get("x-ratelimit-remaining-tokens", ""),

        "x-ratelimit-reset-tokens":
            headers.get("x-ratelimit-reset-tokens", "")
    }


# ==================================================
# SAVE RESPONSE HEADERS
# ==================================================

def save_headers(
    request_id,
    scenario,
    status_code,
    headers
):

    record = {
        "request_id": request_id,
        "scenario": scenario,
        "status_code": status_code,
        **extract_headers(headers)
    }

    with header_lock:

        append_csv(
            HEADERS_FILE,
            HEADER_COLUMNS,
            [record]
        )


# ==================================================
# EXECUTE ONE REAL GROQ REQUEST
# ==================================================

def execute_request(scenario, request_number):

    request_id = (
        f"REAL-{scenario[:3].upper()}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    start_time = datetime.now(
        timezone.utc
    ).isoformat()

    start = time.perf_counter()

    first_token_ms = ""
    total_ms = ""

    tokens_in = ""
    tokens_out = ""

    status = "error"
    status_code = ""

    response_headers = {}
    error_message = ""

    prompt = (
        "Explain the Transformer attention mechanism "
        "in a concise paragraph. Focus on queries, keys, "
        "values, and scaled dot-product attention."
    )

    try:

        # Streaming request.
        # stream_options is intentionally omitted because
        # the installed SDK version does not support it.

        with client.chat.completions.with_streaming_response.create(

            model=MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            stream=True,
            temperature=0.2,
            reasoning_effort="low",
            include_reasoning=False,
            max_completion_tokens=256

        ) as response:

            status_code = response.status_code

            response_headers = dict(
                response.headers
            )

            stream = response.parse()

            for chunk in stream:

                # Capture usage if the API returns it.
                usage = getattr(
                    chunk,
                    "usage",
                    None
                )

                if usage:

                    tokens_in = usage.prompt_tokens
                    tokens_out = usage.completion_tokens

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                content = getattr(
                    delta,
                    "content",
                    None
                )

                # Time to first generated content.
                if content and first_token_ms == "":

                    first_token_ms = round(
                        (
                            time.perf_counter() - start
                        ) * 1000,
                        2
                    )

        total_ms = round(
            (
                time.perf_counter() - start
            ) * 1000,
            2
        )

        status = "ok"

    except APIStatusError as error:

        total_ms = round(
            (
                time.perf_counter() - start
            ) * 1000,
            2
        )

        status_code = error.status_code

        if error.response is not None:

            response_headers = dict(
                error.response.headers
            )

        status = (
            "rate_limited"
            if status_code == 429
            else "error"
        )

        error_message = str(error)

    except Exception as error:

        total_ms = round(
            (
                time.perf_counter() - start
            ) * 1000,
            2
        )

        status = "error"
        error_message = str(error)

    # Save actual response headers.
    save_headers(
        request_id,
        scenario,
        status_code,
        response_headers
    )

    record = {
        "request_id": request_id,
        "scenario": scenario,
        "source": "real",
        "start_time": start_time,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "retrieval_setting": "none",
        "answer_grade": ""
    }

    print(
        f"{request_id} | "
        f"HTTP {status_code} | "
        f"{status} | "
        f"TTFT: {first_token_ms} ms | "
        f"Total: {total_ms} ms"
    )

    if error_message:
        print("Error:", error_message)

    return record


# ==================================================
# STEADY TRAFFIC
# ==================================================

def run_steady():

    scenario = "steady"

    print("\nRunning REAL steady traffic")

    results = []

    for number in range(1, 7):

        result = execute_request(
            scenario,
            number
        )

        results.append(result)

        if number < 6:
            time.sleep(3)

    return results


# ==================================================
# BURST TRAFFIC
# ==================================================

def run_burst():

    scenario = "burst"

    print("\nRunning REAL burst traffic")

    results = []

    with ThreadPoolExecutor(
        max_workers=5
    ) as executor:

        futures = [
            executor.submit(
                execute_request,
                scenario,
                number
            )
            for number in range(1, 6)
        ]

        for future in as_completed(futures):

            try:
                results.append(
                    future.result()
                )

            except Exception as error:
                print("Worker error:", error)

    return results


# ==================================================
# SUSTAINED TRAFFIC
# ==================================================

def run_sustained():

    scenario = "sustained_overload"

    print("\nRunning REAL sustained traffic")

    results = []

    for number in range(1, 13):

        result = execute_request(
            scenario,
            number
        )

        results.append(result)

        if number < 12:
            time.sleep(3)

    return results


# ==================================================
# SUMMARY
# ==================================================

def print_summary(records):

    print("\n" + "=" * 60)
    print("REAL GROQ LOAD TEST SUMMARY")
    print("=" * 60)

    scenarios = sorted(
        set(
            row["scenario"]
            for row in records
        )
    )

    for scenario in scenarios:

        rows = [
            row for row in records
            if row["scenario"] == scenario
        ]

        successful = [
            row for row in rows
            if row["status"] == "ok"
        ]

        rate_limited = sum(
            row["status"] == "rate_limited"
            for row in rows
        )

        errors = sum(
            row["status"] == "error"
            for row in rows
        )

        ttft = [
            float(row["first_token_ms"])
            for row in successful
            if row["first_token_ms"] != ""
        ]

        total = [
            float(row["total_ms"])
            for row in successful
            if row["total_ms"] != ""
        ]

        print(f"\nScenario: {scenario}")
        print("Total requests:", len(rows))
        print("Successful:", len(successful))
        print("Rate limited:", rate_limited)
        print("Errors:", errors)

        if ttft:

            print(
                "Mean first-token latency:",
                f"{sum(ttft) / len(ttft):.2f} ms"
            )

        if total:

            print(
                "Mean total latency:",
                f"{sum(total) / len(total):.2f} ms"
            )


# ==================================================
# EXPERIMENT REGISTRY
# ==================================================

EXPERIMENTS = {
    "steady": run_steady,
    "burst": run_burst,
    "sustained_overload": run_sustained
}


# ==================================================
# MAIN
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="Real Groq API load experiments"
    )

    parser.add_argument(
        "scenario",
        nargs="?",
        default="all",
        choices=[
            "all",
            "steady",
            "burst",
            "sustained_overload"
        ]
    )

    args = parser.parse_args()

    print("=" * 60)
    print("THE RELAY - REAL GROQ EXPERIMENTS")
    print("=" * 60)

    print("Model:", MODEL)
    print("Source: Real Groq API")
    print("Automatic retries: Disabled")
    print("Maximum planned requests: 23")

    if args.scenario == "all":

        selected = EXPERIMENTS.items()

    else:

        selected = [
            (
                args.scenario,
                EXPERIMENTS[args.scenario]
            )
        ]

    all_results = []

    for name, experiment in selected:

        print("\n" + "-" * 60)
        print("Scenario:", name)
        print("-" * 60)

        results = experiment()

        append_csv(
            RESULTS_FILE,
            CSV_COLUMNS,
            results
        )

        all_results.extend(results)

        print(
            f"\nSaved {len(results)} requests."
        )

    print_summary(all_results)

    print("\nExperiments completed.")
    print("Request results:", RESULTS_FILE)
    print("Response headers:", HEADERS_FILE)


if __name__ == "__main__":
    main()
