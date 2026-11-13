
import fitz
import json
from pathlib import Path

PAPERS = [
    {
        "file": "attention_is_all_you_need_v7.pdf",
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "version": 7
    },
    {
        "file": "moshi_v1.pdf",
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "arxiv_id": "2410.00037",
        "version": 1
    },
    {
        "file": "dynamic_chunking_v2.pdf",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "arxiv_id": "2507.07955",
        "version": 2
    }
]

PAPERS_DIR = Path("papers")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

def extract_papers():

    all_pages = []

    for paper in PAPERS:

        pdf_path = PAPERS_DIR / paper["file"]

        if not pdf_path.exists():
            print("Missing:", pdf_path)
            continue

        print(f"\nProcessing: {paper['title']}")

        document = fitz.open(pdf_path)

        print("Total pages:", len(document))

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            record = {
                "paper_title": paper["title"],
                "arxiv_id": paper["arxiv_id"],
                "version": paper["version"],
                "page": page_number,
                "text": text
            }

            all_pages.append(record)

        document.close()

    output_file = OUTPUT_DIR / "paper_pages.jsonl"

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:

        for record in all_pages:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 50)
    print("PDF EXTRACTION COMPLETED")
    print("=" * 50)
    print("Total extracted pages:", len(all_pages))
    print("Output:", output_file)

if __name__ == "__main__":
    extract_papers()
