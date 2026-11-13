
import os
import time

from dotenv import load_dotenv
from groq import Groq


# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY missing from .env"
    )


client = Groq(
    api_key=API_KEY,
    max_retries=0,
    timeout=60.0
)


# ==========================================
# STREAMING GENERATION
# ==========================================

def generate_stream(prompt, max_tokens=1200):

    start = time.perf_counter()

    first_token_ms = None
    output = ""

    tokens_in = None
    tokens_out = None
    finish_reason = None

    stream = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,
        reasoning_effort="low",
        include_reasoning=False,

        max_completion_tokens=max_tokens,
        stream=True
    )

    try:

        for chunk in stream:

            # Capture usage even if the final
            # chunk contains no choices.
            groq_data = getattr(
                chunk,
                "x_groq",
                None
            )

            usage = getattr(
                groq_data,
                "usage",
                None
            )

            if usage:

                tokens_in = getattr(
                    usage,
                    "prompt_tokens",
                    None
                )

                tokens_out = getattr(
                    usage,
                    "completion_tokens",
                    None
                )

            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            delta = choice.delta
            content = delta.content

            if content:

                if first_token_ms is None:

                    first_token_ms = (
                        time.perf_counter() - start
                    ) * 1000

                output += content

            if choice.finish_reason:

                finish_reason = choice.finish_reason

    finally:

        close_stream = getattr(
            stream,
            "close",
            None
        )

        if callable(close_stream):
            close_stream()

    total_ms = (
        time.perf_counter() - start
    ) * 1000

    output = output.strip()

    # Never silently pass an empty answer
    # to the evaluator.

    if not output:

        raise RuntimeError(
            "Groq returned empty answer content. "
            f"Finish reason: {finish_reason}"
        )

    return {
        "text": output,
        "first_token_ms": first_token_ms,
        "total_ms": total_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "finish_reason": finish_reason,
        "model": MODEL
    }
