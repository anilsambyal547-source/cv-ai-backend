"""
Industry detection from CV text using keywords and optional LLM fallback.
"""
import re
import logging
from typing import Optional, List, Dict, Any

import google.genai as genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from config import config

logger = logging.getLogger(__name__)

# Common industry keywords
INDUSTRY_KEYWORDS = {
    "fintech": ["fintech", "financial technology", "banking", "finance", "payments", "insurance", "trading", "blockchain"],
    "healthcare": ["healthcare", "health tech", "medical", "biotech", "pharma", "hospital", "clinical", "health"],
    "ecommerce": ["ecommerce", "e-commerce", "retail", "online shopping", "marketplace", "shopify", "magento"],
    "saas": ["saas", "software as a service", "cloud software", "subscription", "b2b software"],
    "education": ["education", "edtech", "e-learning", "learning management", "training", "academic"],
    "manufacturing": ["manufacturing", "industrial", "supply chain", "logistics", "production", "automation"],
    "media": ["media", "entertainment", "streaming", "publishing", "advertising", "marketing tech"],
    "telecom": ["telecom", "telecommunications", "5g", "networking", "wireless"],
    "automotive": ["automotive", "autonomous vehicles", "electric vehicles", "car", "transportation"],
    "energy": ["energy", "oil & gas", "renewable", "solar", "wind", "utilities"],
    "consulting": ["consulting", "professional services", "management consulting", "strategy"],
    "government": ["government", "public sector", "defense", "military", "public administration"],
}

def detect_industry_rule_based(text: str) -> Optional[str]:
    """Scan CV text for industry keywords, return first matching industry."""
    text_lower = text.lower()
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return industry
    return None

@retry(
    stop=stop_after_attempt(config.EXTRACTION_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def detect_industry_llm(cv_text: str, client) -> Optional[str]:
    """Use Gemini to infer industry from CV text."""
    prompt = f"""
Given the following CV text, identify the most likely industry domain(s) the candidate has experience in.
Return a single industry name from this list: {', '.join(INDUSTRY_KEYWORDS.keys())}.
If unsure, return "unknown". Output only the industry name.

CV:
{cv_text[:2000]}
"""
    try:
        model_name = config.PREFERRED_MODEL or "gemini-2.5-flash"
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=50,
            )
        )
        if response.text:
            industry = response.text.strip().lower()
            if industry in INDUSTRY_KEYWORDS:
                return industry
    except Exception as e:
        logger.error(f"LLM industry detection failed: {e}")
    return None

def detect_industry(cv_text: str) -> Optional[str]:
    """
    Main industry detection: rule‑based first, then LLM fallback.
    """
    # Rule-based
    industry = detect_industry_rule_based(cv_text)
    if industry:
        return industry

    # LLM fallback (only if API available)
    try:
        import google.genai as genai
        client = genai.Client(api_key=config.require_google_api_key())
        industry = detect_industry_llm(cv_text, client)
        if industry:
            logger.info(f"LLM detected industry: {industry}")
            return industry
    except Exception as e:
        logger.warning(f"LLM fallback failed: {e}")

    return None