import fitz  # PyMuPDF

def extract_pdf_text(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_pdf_metadata(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    metadata = doc.metadata
    doc.close()
    return metadata