
import json
import re
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = Path("data/paper_pages.jsonl")
OUTPUT_DIR = Path("data/indexes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OVERLAP = 60

SETTINGS = {
    "A": {
        "chunk_size": 300,
        "top_k": 3,
        "rerank": False
    },
    "B": {
        "chunk_size": 450,
        "top_k": 5,
        "rerank": False
    },
    "C": {
        "chunk_size": 450,
        "top_k": 5,
        "rerank": True
    }
}

# ==========================================
# LOAD MODELS
# ==========================================

print("Loading embedding model...")

embedder = SentenceTransformer(EMBEDDING_MODEL)

tokenizer = AutoTokenizer.from_pretrained(
    EMBEDDING_MODEL
)

reranker = None

# ==========================================
# LOAD EXTRACTED PAPERS
# ==========================================

def load_pages():

    pages = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:
            pages.append(json.loads(line))

    print("Pages loaded:", len(pages))

    return pages


# ==========================================
# TOKEN-BASED CHUNKING
# ==========================================

def create_chunks(pages, chunk_size):

    chunks = []

    for page in pages:

        text = page["text"]

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        start = 0

        while start < len(token_ids):

            end = min(
                start + chunk_size,
                len(token_ids)
            )

            chunk_ids = token_ids[start:end]

            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True
            ).strip()

            if chunk_text:

                chunks.append({
                    "paper_title": page["paper_title"],
                    "arxiv_id": page["arxiv_id"],
                    "version": page["version"],
                    "page": page["page"],
                    "text": chunk_text,
                    "chunk_tokens": len(chunk_ids)
                })

            if end == len(token_ids):
                break

            start = end - OVERLAP

    return chunks


# ==========================================
# EMBEDDING GENERATION
# ==========================================

def generate_embeddings(chunks):

    texts = [
        "Represent this passage for retrieval: " + chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# ==========================================
# BUILD FAISS INDEX
# ==========================================

def build_index(setting_name, config, pages):

    print("\nBuilding setting:", setting_name)

    chunks = create_chunks(
        pages,
        config["chunk_size"]
    )

    print("Chunks created:", len(chunks))

    embeddings = generate_embeddings(chunks)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    # Save index
    index_path = (
        OUTPUT_DIR / f"index_{setting_name}.faiss"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    # Save corresponding metadata
    metadata_path = (
        OUTPUT_DIR / f"metadata_{setting_name}.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("Index saved:", index_path)
    print("Metadata saved:", metadata_path)


# ==========================================
# RETRIEVAL
# ==========================================

def retrieve(query, setting="A"):

    if setting not in SETTINGS:
        raise ValueError("Unknown retrieval setting")

    config = SETTINGS[setting]

    index = faiss.read_index(
        str(OUTPUT_DIR / f"index_{setting}.faiss")
    )

    with (
        OUTPUT_DIR / f"metadata_{setting}.json"
    ).open("r", encoding="utf-8") as file:

        chunks = json.load(file)

    query_embedding = embedder.encode(
        ["Represent this question for retrieving relevant passages: " + query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    # Retrieve extra candidates for reranking.
    candidate_count = (
        min(15, len(chunks))
        if config["rerank"]
        else config["top_k"]
    )

    scores, indices = index.search(
        query_embedding,
        candidate_count
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        result = chunks[int(idx)].copy()
        result["similarity_score"] = float(score)

        results.append(result)

    # ======================================
    # OPTIONAL CROSS-ENCODER RERANKING
    # ======================================

    if config["rerank"]:

        global reranker

        if reranker is None:
            print("Loading reranker...")
            reranker = CrossEncoder(
                RERANKER_MODEL,
                max_length=512
            )

        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = reranker.predict(pairs)

        for result, score in zip(results, scores):
            result["rerank_score"] = float(score)

        results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

    return results[:config["top_k"]]


# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_results(results):

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 55)
        print("Passage:", i)
        print("Paper:", result["paper_title"])
        print("arXiv:", result["arxiv_id"])
        print("Version:", result["version"])
        print("Page:", result["page"])
        print("Tokens:", result["chunk_tokens"])
        print(
            "Similarity:",
            round(result["similarity_score"], 4)
        )

        if "rerank_score" in result:
            print(
                "Rerank score:",
                round(result["rerank_score"], 4)
            )

        print("\nText:\n")
        print(result["text"][:1000])


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("=" * 55)
    print("THE RELAY - RETRIEVAL ENGINE")
    print("=" * 55)

    pages = load_pages()

    # Build all three experimental configurations.
    for name, config in SETTINGS.items():

        build_index(
            name,
            config,
            pages
        )

    print("\nAll three indexes created.")

    # Retrieval smoke test.
    question = (
        "How does multi-head attention work "
        "in the Transformer architecture?"
    )

    print("\nTesting retrieval:")
    print(question)

    for setting in SETTINGS:

        print("\n\nTESTING SETTING:", setting)

        results = retrieve(
            question,
            setting
        )

        display_results(results)

    print("\nRetrieval engine ready.")
