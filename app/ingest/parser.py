"""
Extract text from PDFs, with OCR fallback for scanned pages.
Includes functions to split CVs by page and by headers.
"""
import fitz
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def extract_text_from_pdf(path: str) -> str:
    """Extract all text from a PDF, falling back to OCR when needed."""
    doc = fitz.open(path)
    full_text = []

    try:
        tesseract_available = fitz.TesseractAvailable()
    except AttributeError:
        tesseract_available = False

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text and len(text.strip()) > 100:
            full_text.append(text)
            continue

        if tesseract_available:
            try:
                ocr_text = page.get_text("ocr")
                if ocr_text and ocr_text.strip():
                    full_text.append(ocr_text)
                    logger.info(f"Page {page_num} processed with OCR.")
                else:
                    full_text.append("")
            except Exception as e:
                logger.warning(f"OCR failed on page {page_num}: {e}")
                full_text.append(text or "")
        else:
            logger.warning(
                f"Page {page_num} may contain images but Tesseract not available. "
                "Install Tesseract OCR and ensure it is in PATH."
            )
            full_text.append(text or "")

    return "\n".join(full_text).strip()

def extract_cvs_from_pdf(path: str) -> List[str]:
    """
    Return one string per page, assuming each page is a separate CV.
    Can be enhanced later to detect multi‑page CVs.
    """
    doc = fitz.open(path)
    cvs = []

    try:
        tesseract_available = fitz.TesseractAvailable()
    except AttributeError:
        tesseract_available = False

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text and len(text.strip()) > 100:
            cvs.append(text.strip())
            continue

        if tesseract_available:
            try:
                ocr_text = page.get_text("ocr")
                if ocr_text and ocr_text.strip():
                    cvs.append(ocr_text.strip())
                    logger.info(f"Page {page_num} processed with OCR.")
                else:
                    cvs.append("")
            except Exception as e:
                logger.warning(f"OCR failed on page {page_num}: {e}")
                cvs.append(text.strip() if text else "")
        else:
            logger.warning(f"Page {page_num} may be image-based but Tesseract missing.")
            cvs.append(text.strip() if text else "")

    cvs = [cv for cv in cvs if cv]
    logger.info(f"Extracted {len(cvs)} CVs from {path}")
    return cvs

def split_cvs_by_header(text: str) -> Dict[str, str]:
    """
    Split CV text into sections based on common headers.
    Returns a dictionary mapping header names to section content.
    This is used by the extractor for potential future section-aware parsing.
    """
    headers = [
        "SUMMARY", "PROFILE", "OBJECTIVE",
        "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT",
        "EDUCATION", "ACADEMIC BACKGROUND",
        "SKILLS", "TECHNICAL SKILLS", "CORE COMPETENCIES",
        "PROJECTS", "PROJECT EXPERIENCE",
        "CERTIFICATIONS", "LICENSES",
        "LANGUAGES", "LANGUAGE SKILLS",
        "PUBLICATIONS", "AWARDS", "HONORS",
        "VOLUNTEERING", "VOLUNTEER WORK",
        "REFERENCES"
    ]
    pattern = re.compile(
        r'^(?:' + '|'.join(re.escape(h) for h in headers) + r')\s*:?\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    lines = text.splitlines()
    sections = {}
    current_header = None
    current_content = []
    for line in lines:
        stripped = line.strip()
        if pattern.match(stripped):
            if current_header is not None:
                sections[current_header] = "\n".join(current_content).strip()
            current_header = stripped.rstrip(':').strip().upper()
            current_content = []
        else:
            if current_header is not None:
                current_content.append(line)
    if current_header is not None:
        sections[current_header] = "\n".join(current_content).strip()
    return sections