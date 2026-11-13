
import argparse
import csv
import sys
import time
import uuid

from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# PROJECT PATHS AND IMPORTS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from simulator import SimulatedModelService
from rate_limiter import RateLimiter, RateLimitExceeded


# ==========================================
# CONFIGURATION
# ==========================================

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "results.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIG = {
    "base_latency_ms": 250,
    "latency_jitter_ms": 80,
    "failure_rate": 0.0,
    "overload_threshold": 5,
    "overload_slowdown_ms": 150,
    "chunk_size": 24,
    "seed": 42
}

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
# REQUEST EXECUTION
# ==========================================

def execute_request(
    service,
    limiter,
    scenario,
    request_number
):

    request_id = (
        f"{scenario[:3].upper()}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    start_time = datetime.now(
        timezone.utc
    ).isoformat()

    start = time.perf_counter()

    first_token_ms = ""
    total_ms = ""
    status = "error"

    try:

        # Check rate limits before calling the service.
        limiter.acquire()

        response_parts = []

        for chunk in service.generate_stream(
            prompt="Explain the Transformer attention mechanism.",
            request_id=request_id
        ):

            if first_token_ms == "":

                first_token_ms = round(
                    (time.perf_counter() - start) * 1000,
                    2
                )

            response_parts.append(chunk)

        total_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        status = "ok"

    except RateLimitExceeded as error:

        status = "rate_limited"

        total_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

    except Exception as error:

        status = "error"

        total_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        print(
            f"Request {request_id} failed: {error}"
        )

    return {
        "request_id": request_id,
        "scenario": scenario,
        "source": "simulated",
        "start_time": start_time,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "status": status,
        "retries": 0,
        "tokens_in": "",
        "tokens_out": "",
        "retrieval_setting": "B",
        "answer_grade": ""
    }


# ==========================================
# RUN CONCURRENT REQUESTS
# ==========================================

def run_concurrent_batch(
    service,
    limiter,
    scenario,
    request_count,
    max_workers
):

    results = []

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        futures = [
            executor.submit(
                execute_request,
                service,
                limiter,
                scenario,
                number
            )
            for number in range(
                1,
                request_count + 1
            )
        ]

        for future in as_completed(futures):
            results.append(future.result())

    return results


# ==========================================
# STEADY TRAFFIC
# ==========================================

def steady_traffic():

    scenario = "steady"

    print("\nRunning steady traffic...")

    service = SimulatedModelService(**MODEL_CONFIG)

    limiter = RateLimiter(
        rpm_limit=100,
        daily_limit=1000
    )

    results = []

    for number in range(1, 11):

        result = execute_request(
            service,
            limiter,
            scenario,
            number
        )

        results.append(result)

        print(
            f"Request {number}/10: {result['status']}"
        )

        time.sleep(0.5)

    print("Service statistics:", service.get_stats())

    return results


# ==========================================
# BURST TRAFFIC
# ==========================================

def burst_traffic():

    scenario = "burst"

    print("\nRunning burst traffic...")

    service = SimulatedModelService(**MODEL_CONFIG)

    limiter = RateLimiter(
        rpm_limit=100,
        daily_limit=1000
    )

    results = run_concurrent_batch(
        service,
        limiter,
        scenario,
        request_count=20,
        max_workers=20
    )

    print("Service statistics:", service.get_stats())

    return results


# ==========================================
# CONTINUOUS SUSTAINED OVERLOAD
# ==========================================

def sustained_overload():

    scenario = "sustained_overload"

    print("\nRunning continuous sustained overload...")

    service = SimulatedModelService(**MODEL_CONFIG)

    # High limits prevent rate limiting from interfering
    # with this experiment.
    limiter = RateLimiter(
        rpm_limit=10000,
        daily_limit=10000
    )

    results = []
    futures = []

    duration_seconds = 15
    arrival_interval_seconds = 0.1
    max_workers = 12

    start_time = time.perf_counter()
    request_number = 0

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        while (
            time.perf_counter() - start_time
            < duration_seconds
        ):

            request_number += 1

            futures.append(
                executor.submit(
                    execute_request,
                    service,
                    limiter,
                    scenario,
                    request_number
                )
            )

            time.sleep(arrival_interval_seconds)

        print(
            f"\nSubmitted {request_number} requests."
        )

        print("Waiting for active requests to finish...")

        for future in as_completed(futures):
            results.append(future.result())

    print("Service statistics:", service.get_stats())

    return results


# ==========================================
# HALVED RPM LIMIT
# ==========================================

def halved_rpm_limit():

    scenario = "halved_rpm"

    print("\nRunning halved RPM limit experiment...")

    service = SimulatedModelService(**MODEL_CONFIG)

    # Normal experimental limit: 30 RPM.
    # Halved experimental limit: 15 RPM.
    limiter = RateLimiter(
        rpm_limit=15,
        daily_limit=1000
    )

    results = run_concurrent_batch(
        service,
        limiter,
        scenario,
        request_count=20,
        max_workers=20
    )

    print("Rate limiter statistics:", limiter.get_stats())

    return results


# ==========================================
# DAILY BUDGET EXHAUSTION
# ==========================================

def exhausted_daily_budget():

    scenario = "daily_budget_exhausted"

    print("\nRunning daily budget exhaustion experiment...")

    service = SimulatedModelService(**MODEL_CONFIG)

    limiter = RateLimiter(
        rpm_limit=100,
        daily_limit=5
    )

    results = run_concurrent_batch(
        service,
        limiter,
        scenario,
        request_count=10,
        max_workers=10
    )

    print("Rate limiter statistics:", limiter.get_stats())

    return results


# ==========================================
# SAVE RESULTS
# ==========================================

def save_results(records):

    if not records:
        print("No records to save.")
        return

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

        writer.writerows(records)

    print("\nResults saved:", OUTPUT_FILE)


# ==========================================
# PRINT SUMMARY
# ==========================================

def print_summary(records):

    print("\n" + "=" * 65)
    print("LOAD TEST SUMMARY")
    print("=" * 65)

    scenarios = sorted(
        set(
            record["scenario"]
            for record in records
        )
    )

    for scenario in scenarios:

        rows = [
            record
            for record in records
            if record["scenario"] == scenario
        ]

        successful = sum(
            row["status"] == "ok"
            for row in rows
        )

        rate_limited = sum(
            row["status"] == "rate_limited"
            for row in rows
        )

        errors = sum(
            row["status"] == "error"
            for row in rows
        )

        successful_times = [
            float(row["total_ms"])
            for row in rows
            if row["status"] == "ok"
            and row["total_ms"] != ""
        ]

        successful_ttft = [
            float(row["first_token_ms"])
            for row in rows
            if row["status"] == "ok"
            and row["first_token_ms"] != ""
        ]

        mean_latency = (
            sum(successful_times) / len(successful_times)
            if successful_times
            else 0
        )

        mean_ttft = (
            sum(successful_ttft) / len(successful_ttft)
            if successful_ttft
            else 0
        )

        print(f"\nScenario: {scenario}")
        print("Total requests:", len(rows))
        print("Successful:", successful)
        print("Rate limited:", rate_limited)
        print("Errors:", errors)

        print(
            f"Mean first-token latency: "
            f"{mean_ttft:.2f} ms"
        )

        print(
            f"Mean total latency: "
            f"{mean_latency:.2f} ms"
        )


# ==========================================
# EXPERIMENT REGISTRY
# ==========================================

EXPERIMENTS = {
    "steady": steady_traffic,
    "burst": burst_traffic,
    "sustained_overload": sustained_overload,
    "halved_rpm": halved_rpm_limit,
    "daily_budget_exhausted": exhausted_daily_budget
}


# ==========================================
# MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser(
        description="The Relay simulated load experiments"
    )

    parser.add_argument(
        "scenario",
        nargs="?",
        default="all",
        choices=[
            "all",
            "steady",
            "burst",
            "sustained_overload",
            "halved_rpm",
            "daily_budget_exhausted"
        ],
        help="Choose one experiment or run all experiments."
    )

    args = parser.parse_args()

    print("=" * 65)
    print("THE RELAY - LOAD TEST EXPERIMENTS")
    print("=" * 65)

    all_results = []

    if args.scenario == "all":

        selected_experiments = EXPERIMENTS.items()

    else:

        selected_experiments = [
            (
                args.scenario,
                EXPERIMENTS[args.scenario]
            )
        ]

    for name, experiment in selected_experiments:

        print("\n" + "-" * 65)
        print("Scenario:", name)
        print("-" * 65)

        results = experiment()

        all_results.extend(results)

        print(
            f"\nCompleted {len(results)} requests."
        )

        # Save each scenario immediately so results
        # are preserved even if a later scenario fails.
        save_results(results)

    print_summary(all_results)

    print("\nSelected experiments completed.")
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
