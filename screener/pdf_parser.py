import re

import pymupdf  #pip install pymupdf


class ResumeReadError(Exception):
    """Raised when a resume cannot be read or contains no extractable text."""


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(name: str, data: bytes) -> str:
    try:
        if name.lower().endswith(".pdf"):
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                text = "\n".join(page.get_text("text") for page in doc)
        else:
            text = data.decode("utf-8", errors="ignore")
    except Exception as exc:  # corrupted / encrypted file
        raise ResumeReadError(f"Could not open file: {exc}") from exc

    text = clean_text(text)
    if len(text) < 50:
        raise ResumeReadError("No readable text found (scanned PDF? OCR is needed).")
    return text
