import fitz  # PyMuPDF
import traceback

def extract_pdf_text(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF file given as bytes.
    Returns combined text from all pages.
    """
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = []
            for page in doc:
                text = page.get_text("text")
                if text:
                    full_text.append(text.strip())
            return "\n\n".join(full_text)
    except Exception as e:
        print(f"[ERROR] Failed to extract text from PDF: {e}")
        traceback.print_exc()
        return ""


def extract_pdf_metadata(pdf_bytes: bytes) -> dict:
    """
    Extract metadata from a PDF (title, author, creation date, etc.).
    Returns a cleaned dictionary (filters empty or None fields).
    """
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            raw_meta = doc.metadata or {}
            metadata = {
                k: v for k, v in raw_meta.items()
                if v and v.strip() and v.lower() not in ["", "none", "null"]
            }
            metadata["page_count"] = doc.page_count
            return metadata
    except Exception as e:
        print(f"[ERROR] Failed to extract metadata from PDF: {e}")
        traceback.print_exc()
        return {}


def extract_pdf_pages(pdf_bytes: bytes) -> list:
    """
    Optionally extract text from each page as a list (for detailed analysis).
    """
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            pages = []
            for i, page in enumerate(doc):
                pages.append({
                    "page_number": i + 1,
                    "text": page.get_text("text").strip()
                })
            return pages
    except Exception as e:
        print(f"[ERROR] Failed to extract pages from PDF: {e}")
        traceback.print_exc()
        return []
