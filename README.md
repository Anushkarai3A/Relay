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

| Component | Technology
