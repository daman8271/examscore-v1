import os
import sys
import argparse
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv
from utils import embed_text, get_pinecone_index

load_dotenv()


def extract_pages_ocr(pdf_path: str) -> list[dict]:
    """Convert each PDF page to an image and OCR it."""
    print("Converting PDF pages to images (this may take a moment)...")
    images = convert_from_path(pdf_path, dpi=200)
    total = len(images)
    print(f"Found {total} pages. Starting OCR...\n")

    pages = []
    for i, image in enumerate(images):
        page_num = i + 1
        print(f"OCR: page {page_num}/{total}...", end="\r", flush=True)
        text = pytesseract.image_to_string(image)
        if text and text.strip():
            pages.append({"page_number": page_num, "text": text.strip()})

    print(f"\nOCR complete. {len(pages)}/{total} pages had extractable text.\n")
    return pages


def ingest_pdf(pdf_path: str, batch_size: int = 50):
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Reading PDF: {pdf_path}")
    pages = extract_pages_ocr(pdf_path)
    total = len(pages)

    if total == 0:
        print("No text found after OCR. Exiting.")
        sys.exit(1)

    index = get_pinecone_index()

    # Process in batches to avoid memory issues on large PDFs
    for batch_start in range(0, total, batch_size):
        batch = pages[batch_start : batch_start + batch_size]
        vectors = []

        for page in batch:
            page_num = page["page_number"]
            print(f"Embedding page {page_num} of {total}...", end="\r", flush=True)

            embedding = embed_text(page["text"])
            vectors.append({
                "id": f"page-{page_num}",
                "values": embedding,
                "metadata": {
                    "page_number": page_num,
                    "text": page["text"],
                    "source": os.path.basename(pdf_path),
                },
            })

        index.upsert(vectors=vectors)
        last_page = batch[-1]["page_number"]
        print(f"Uploaded pages {batch[0]['page_number']}–{last_page} of {total}   ")

    print(f"\nDone! {total} pages indexed in Pinecone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into Pinecone.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    args = parser.parse_args()
    ingest_pdf(args.pdf_path)
