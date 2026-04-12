# ingestion.py
import time
import logging
from pathlib import Path
from typing import List
import tqdm

from config import config
from parser import extract_text_from_pdf
from extractor import extract_structured_data
from indexer import index_cv

logger = logging.getLogger(__name__)

def batch_ingest(pdf_paths: List[Path], batch_size: int = 10, rate_limit: float = 2.0) -> int:
    """
    Process PDFs in batches with rate limiting.
    Returns total number of CVs indexed.
    """
    total_indexed = 0
    for i in range(0, len(pdf_paths), batch_size):
        batch = pdf_paths[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(pdf_paths)-1)//batch_size + 1}")
        for pdf in tqdm.tqdm(batch, desc=f"Batch {i//batch_size + 1}"):
            try:
                text = extract_text_from_pdf(str(pdf))
                data = extract_structured_data(text)
                index_cv(pdf.stem, data)
                total_indexed += len(data)
            except Exception as e:
                logger.error(f"Failed to process {pdf.name}: {e}")
        if i + batch_size < len(pdf_paths):
            logger.info(f"Rate limit: waiting {rate_limit}s...")
            time.sleep(rate_limit)
    return total_indexed