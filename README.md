# The Relay — Research Assistant & Load Testing Framework

**A Python-based research assistant that retrieves evidence from research papers, generates cited answers using Groq, validates citations, and evaluates performance under different load conditions.**

## Overview

The Relay is an evidence-grounded research assistant designed to answer questions from a fixed collection of research papers while measuring the performance and limitations of a hosted language model service.

It combines document processing, semantic retrieval, answer generation, citation validation, and load testing in a single pipeline.

The project also includes a simulated model service and rate limiter to study how system performance changes under steady traffic, burst requests, sustained load, and restricted request budgets.

## Key Features

- **PDF Processing:** Extracts text from research papers while preserving page-level references.
- **Semantic Retrieval:** Uses BGE embeddings and FAISS for similarity-based passage retrieval.
- **Cross-Encoder Reranking:** Supports reranking retrieved passages to improve relevance.
- **LLM Integration:** Uses Groq API with `openai/gpt-oss-20b`.
- **Citation Validation:** Checks whether generated citations correspond to retrieved evidence.
- **Evidence-Based Refusal:** Avoids generating unsupported answers when sufficient evidence is unavailable.
- **Performance Monitoring:** Measures time to first token (TTFT), total response time, and token usage where available.
- **Load Testing:** Evaluates steady, burst, and sustained traffic.
- **Service Simulation:** Simulates latency, failures, concurrency slowdown, streaming, and rate limits.
- **Rate Limiting:** Implements rolling-window per-minute limits and daily request budgets.

## Research Paper Collection

The system works with these specific arXiv versions:

| Research Paper | arXiv Version |
|---|---|
| Attention Is All You Need | 1706.03762 v7 |
| Moshi: a Speech-Text Foundation Model for Real-Time Dialogue | 2410.00037 v1 |
| Dynamic Chunking for End-to-End Hierarchical Sequence Modeling | 2507.07955 v2 |

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Language Model | Groq — openai/gpt-oss-20b |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Vector Search | FAISS |
| Reranking | Cross-Encoder |
| PDF Processing | PyMuPDF |
| Data Analysis | Pandas |
| API Integration | Groq Python SDK |
| Configuration | Python dotenv |

## System Architecture

```text
             Research Papers (PDF)
                      |
                      v
              PDF Text Extraction
                      |
                      v
               Text Chunking
                      |
                      v
             BGE Embeddings
                      |
                      v
                FAISS Index
                      |
              User Question
                      |
                      v
             Semantic Retrieval
                      |
                      v
             Optional Reranking
                      |
                      v
                Groq LLM
                      |
                      v
             Citation Validation
                      |
              +-------+-------+
              |               |
          Valid Evidence   Invalid Evidence
              |               |
              v               v
         Final Answer    Correction / Refusal
```

The load-testing system operates separately to evaluate the hosted model and its local stand-in under different traffic conditions.

## Retrieval Experiments

Three retrieval configurations were evaluated:

| Setting | Chunk Size | Top-K | Reranking |
|---|---:|---:|---|
| A | 300 | 3 | No |
| B | 450 | 5 | No |
| C | 450 | 5 | Yes |

Setting B was selected provisionally because it had the lowest measured mean retrieval time among the three configurations without the additional reranking cost.

The retrieval evaluation covered 15 benchmark questions, including answerable and deliberately unanswerable questions.

## Load Testing

The project evaluates the following scenarios:

1. **Steady Traffic:** Sequential requests under normal traffic.
2. **Burst Traffic:** Multiple concurrent requests.
3. **Sustained Overload:** Repeated requests over an extended period.
4. **Halved RPM:** Simulation with a reduced requests-per-minute limit.
5. **Daily Budget Exhaustion:** Simulation of an exhausted daily request budget.

Real Groq API experiments were performed for steady, burst, and sustained traffic with automatic retries disabled.

The remaining quota-exhaustion scenarios were tested using the local simulator rather than deliberately exhausting the real API quota.

## Performance Metrics

The experiments record:

- Time to First Token (TTFT)
- Total response latency
- HTTP status and request outcome
- Retry count
- Input and output tokens, when available
- Retrieval configuration
- Answer grade, when manually evaluated

Raw results are stored in CSV format for further analysis.

## Project Structure

```text
the-relay/
│
├── src/
│   ├── pdf_processor.py
│   ├── retrieval.py
│   ├── model_client.py
│   ├── pipeline.py
│   ├── evaluator.py
│   └── rate_limiter.py
│
├── experiments/
│   ├── load_test.py
│   ├── real_load_test.py
│   ├── baseline.py
│   └── test_groq.py
│
├── papers/
│   └── Research paper PDFs
│
├── data/
│   └── Extracted paper text and indexes
│
├── results/
│   ├── results.csv
│   ├── retrieval_evaluation.csv
│   └── retrieval_summary.json
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

*The structure above describes the main project components; adjust filenames to match the final repository.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd the-relay
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```powershell
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Never commit your actual API key.

### 5. Prepare the research papers

Place the required PDF versions inside the `papers/` directory and run the PDF processing script.

```bash
python src/pdf_processor.py
```

### 6. Run the research assistant

Run the pipeline using the entry point configured in your project.

## Running Experiments

Run simulated load tests:

```bash
python experiments/load_test.py all
```

Run real Groq experiments:

```bash
python experiments/real_load_test.py steady
python experiments/real_load_test.py burst
python experiments/real_load_test.py sustained_overload
```

Real API experiments require a valid Groq API key and consume API quota.

## Limitations

- Real API rate-limit exhaustion and recovery behaviour were not experimentally verified.
- The simulator approximates service behaviour and does not reproduce Groq's internal infrastructure.
- Citation validation checks references against retrieved passages; it does not independently prove semantic entailment.
- Retrieval rank does not guarantee that a passage contains sufficient evidence to answer a question.
- Some answer-quality and token-cost measurements remain unmeasured.

## Future Improvements

- Improve semantic verification of generated claims.
- Expand benchmark questions and manual answer grading.
- Add queue-aware admission control.
- Evaluate recovery behaviour using controlled service failures.
- Improve latency and token-cost reporting.
- Add support for additional document collections and model providers.

## Author

**Anushka Maneesh Rai**

Computer Science and Engineering

### Project Focus

Research Assistant Systems | Retrieval-Augmented Generation | LLM Evaluation | Performance Engineering | Rate Limiting

---

*The Relay explores the relationship between retrieval quality, evidence-grounded generation, and the performance constraints of hosted language model services.*
