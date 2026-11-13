
import httpx
from pathlib import Path

PAPERS_DIR = Path("papers")
PAPERS_DIR.mkdir(exist_ok=True)

PAPERS = [
    {
        "name": "attention_is_all_you_need_v7.pdf",
        "url": "https://arxiv.org/pdf/1706.03762v7",
        "version": "1706.03762v7"
    },
    {
        "name": "moshi_v1.pdf",
        "url": "https://arxiv.org/pdf/2410.00037v1",
        "version": "2410.00037v1"
    },
    {
        "name": "dynamic_chunking_v2.pdf",
        "url": "https://arxiv.org/pdf/2507.07955v2",
        "version": "2507.07955v2"
    }
]

def download_paper(paper):

    destination = PAPERS_DIR / paper["name"]

    print(f"\nDownloading {paper['version']}...")

    try:
        with httpx.Client(
            timeout=60,
            follow_redirects=True
        ) as client:

            response = client.get(paper["url"])
            response.raise_for_status()

            content = response.content

            if not content.startswith(b"%PDF"):
                raise ValueError("Downloaded file is not a PDF")

            destination.write_bytes(content)

            print("Downloaded:", destination)
            print("Size:", round(len(content) / 1024, 2), "KB")
            print("Status: SUCCESS")

    except Exception as exc:
        print("Download failed:", paper["version"])
        print("Error:", str(exc))


if __name__ == "__main__":

    print("=" * 50)
    print("THE RELAY - RESEARCH PAPER DOWNLOAD")
    print("=" * 50)

    for paper in PAPERS:
        download_paper(paper)

    print("\nDownload process completed.")
