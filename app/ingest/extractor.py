"""
Hybrid CV extractor: rule-based + LLM.
Optimized for extreme memory efficiency on Render.
"""
import re
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime

import google.genai as genai
from google.genai import types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from config import config
from .parser import extract_cvs_from_pdf, split_cvs_by_header
from app.schemas.cv import CVSchema, WorkExperience, parse_date, compute_experience_from_work, classify_education_level, classify_seniority
from app.industry_detector import detect_industry

logger = logging.getLogger(__name__)

# Load spaCy model only if available (Extreme Lazy Loading)
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            print("Loading spaCy model 'en_core_web_sm'...")
            _nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError, Exception):
            # Silent fallback – use deterministic logic only
            logger.info("spaCy fallback: Using deterministic extraction.")
            return None
    return _nlp

def extract_name_with_spacy(text: str) -> str:
    nlp = get_nlp()
    if nlp is None:
        return ""
    doc = nlp(text[:2000])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return ""

def extract_name_fallback(text: str) -> str:
    lines = text.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w):
            return line
    return ""

def extract_name(text: str) -> str:
    name = extract_name_with_spacy(text)
    if not name:
        name = extract_name_fallback(text)
    return name

def normalize_skill(raw_skill: str) -> str:
    skill_lower = raw_skill.lower().strip()
    for canon, variants in config.SKILL_SYNONYMS.items():
        if skill_lower == canon or skill_lower in variants:
            return canon
    return raw_skill.strip().capitalize()

# ... (rest of the file remains same, keeping imports clean)
# ----------------------------------------------------------------------
# Gemini model selection (cached)
# ----------------------------------------------------------------------
_WORKING_MODEL = None

def get_working_model(client) -> str:
    global _WORKING_MODEL
    if _WORKING_MODEL is not None:
        return _WORKING_MODEL

    if config.PREFERRED_MODEL:
        try:
            client.models.get(model=config.PREFERRED_MODEL)
            logger.info(f"✅ Using preferred model: {config.PREFERRED_MODEL}")
            _WORKING_MODEL = config.PREFERRED_MODEL
            return _WORKING_MODEL
        except Exception:
            logger.warning(f"Preferred model {config.PREFERRED_MODEL} not available, scanning...")

    try:
        models = client.models.list()
        priority = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp",
            "gemini-2.0-pro-exp",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-pro-preview-03-25",
        ]
        for model_id in priority:
            try:
                m = client.models.get(model=model_id)
                if "generateContent" in m.supported_actions:
                    logger.info(f"✅ Using model: {model_id}")
                    _WORKING_MODEL = model_id
                    return _WORKING_MODEL
            except Exception:
                continue

        for m in models:
            name = m.name.replace("models/", "")
            if any(x in name for x in ["audio", "image", "vision", "tts", "er-"]):
                continue
            if "generateContent" in m.supported_actions:
                logger.info(f"✅ Using fallback model: {name}")
                _WORKING_MODEL = name
                return _WORKING_MODEL
    except Exception as e:
        logger.error(f"Model discovery error: {e}")

    raise RuntimeError("❌ No suitable Gemini model found. Check API key and quota.")

# ----------------------------------------------------------------------
# LLM extraction with JSON mode
# ----------------------------------------------------------------------
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "years_experience": {"type": "number"},
        "summary": {"type": "string"},
        "work_experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "job_title": {"type": "string"},
                    "company": {"type": "string"},
                    "start_date": {"type": "string"},
                    "end_date": {"type": "string"},
                    "description": {"type": "string"}
                }
            }
        },
        "education": {"type": "array", "items": {"type": "string"}},
        "certifications": {"type": "array", "items": {"type": "string"}},
        "location": {"type": "string"},
        "remote_available": {"type": "boolean"}
    },
    "required": ["name", "email", "skills", "years_experience"]
}

@retry(
    stop=stop_after_attempt(config.EXTRACTION_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def _call_gemini(client, model_name: str, cv_text: str) -> Optional[Dict[str, Any]]:
    prompt = f"""
You are an expert resume parser. Extract the following information from the resume text below and return **valid JSON** matching this schema:

{json.dumps(EXTRACTION_SCHEMA, indent=2)}

- If a field is missing, use an empty string or empty list.
- For "years_experience", estimate total years of professional experience (as a number, can be 0).
- "skills" should be a list of strings.
- "work_experience" should be a list of objects, each with job_title, company, start_date, end_date, and description. Use full month names or YYYY-MM format if possible.
- "education" should be a list of strings (e.g., "BSc Computer Science, University of X, 2015-2019").
- "location" should be the candidate's city or region.
- "remote_available" should be true if the candidate mentions remote work preference.
- Return only the JSON object, no extra text.

Resume:
{cv_text[:3500]}
"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=config.EXTRACTION_MAX_TOKENS,
                response_mime_type="application/json",
            )
        )
        if not response.candidates:
            logger.warning("No candidates in response")
            return None
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            logger.warning("Empty content")
            return None
        text = "".join(part.text for part in candidate.content.parts if hasattr(part, "text"))
        if not text:
            logger.warning("No text in response")
            return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}")
            cleaned = text.strip().strip('`').strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.error(f"Could not parse JSON even after cleaning: {text[:200]}...")
                return None

        required = EXTRACTION_SCHEMA["required"]
        if not all(k in data for k in required):
            logger.warning(f"Missing required fields: {set(required) - set(data.keys())}")
            return None

        return data
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        raise

# ----------------------------------------------------------------------
# Simplified regex helpers for robust fallback
# ----------------------------------------------------------------------
def extract_email(text: str) -> str:
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0).lower() if match else ""

def extract_phone(text: str) -> str:
    match = re.search(r'(\+?\d[\d\-\s\(\)]{7,}\d)', text)
    return match.group(0).strip() if match else ""

# ----------------------------------------------------------------------
# Repair functions
# ----------------------------------------------------------------------
COMMON_CITIES = [
    "new york", "san francisco", "london", "berlin", "paris", "tokyo",
    "bangalore", "mumbai", "delhi", "hyderabad", "pune", "chennai",
    "seattle", "austin", "boston", "chicago", "los angeles", "toronto",
    "sydney", "melbourne", "dublin", "amsterdam", "singapore"
]

def repair_email(text: str, current: str) -> str:
    if current:
        return current
    return extract_email(text)

def repair_phone(text: str, current: str) -> str:
    if current:
        return current
    return extract_phone(text)

def repair_name(text: str, current: str) -> str:
    if current:
        return current
    return extract_name(text)

def repair_skills(text: str, current_skills: List[str]) -> List[str]:
    text_lower = text.lower()
    found = set(current_skills)
    for canon, variants in config.SKILL_SYNONYMS.items():
        if canon in text_lower or any(var in text_lower for var in variants):
            found.add(canon)
    return sorted(found)

def repair_experience(text: str, current: float) -> float:
    if current > 0:
        return current
    patterns = [
        r'(\d+)[\+]?\s*(?:years?|yrs?)',
        r'experience\s*(?:of|:)?\s*(\d+)[\+]?\s*(?:years?|yrs?)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return 0.0

def repair_work_experience(text: str, current: List[Dict]) -> List[Dict]:
    if current and len(current) > 0:
        return current
    job_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:at|@|,)\s+([A-Z][a-zA-Z0-9\s]+)'
    matches = re.findall(job_pattern, text)
    if matches:
        return [{"job_title": m[0], "company": m[1], "start_date": "", "end_date": "", "description": ""} for m in matches]
    return []

def repair_education_level(text: str, current: Optional[str], education_list: List[str]) -> Optional[str]:
    if current:
        return current
    if education_list:
        return classify_education_level(education_list)
    return classify_education_level([text[:1000]])

def repair_location(text: str, current: Optional[str]) -> Optional[str]:
    if current:
        return current
    loc_patterns = [
        r'(?:location|city|based in|resides? in)\s*:?\s*([A-Za-z\s,]+)',
        r'from\s+([A-Za-z\s,]+)',
    ]
    for pat in loc_patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    text_lower = text.lower()
    for city in COMMON_CITIES:
        if city in text_lower:
            return city.title()
    return None

def repair_remote_available(text: str, current: Optional[bool]) -> Optional[bool]:
    if current is not None:
        return current
    remote_keywords = [
        r'\bremote\b', r'\bwork from home\b', r'\bwfh\b', r'\bhybrid\b',
        r'\btelecommute\b', r'\bvirtual\b'
    ]
    for kw in remote_keywords:
        if re.search(kw, text, re.IGNORECASE):
            return True
    return None

def repair_seniority(work_experience: List[Dict], summary: str, current: Optional[str]) -> Optional[str]:
    if current:
        return current
    job_titles = [w.get("job_title", "") for w in work_experience if w.get("job_title")]
    return classify_seniority(job_titles, summary)

def repair_record(record: Dict, raw_text: str, deterministic: Dict) -> Dict:
    record["name"] = repair_name(raw_text, record.get("name", ""))
    record["email"] = repair_email(raw_text, record.get("email", ""))
    record["phone"] = repair_phone(raw_text, record.get("phone", ""))
    record["skills"] = repair_skills(raw_text, record.get("skills", []))
    record["years_experience"] = repair_experience(raw_text, record.get("years_experience", 0.0))
    record["work_experience"] = repair_work_experience(raw_text, record.get("work_experience", []))
    record["education_level"] = repair_education_level(
        raw_text,
        record.get("education_level"),
        record.get("education", [])
    )
    record["location"] = repair_location(raw_text, record.get("location"))
    record["remote_available"] = repair_remote_available(raw_text, record.get("remote_available"))
    record["seniority"] = repair_seniority(
        record.get("work_experience", []),
        record.get("summary", ""),
        record.get("seniority")
    )
    return record

# ----------------------------------------------------------------------
# Validation + repair + confidence + industry detection
# ----------------------------------------------------------------------
def validate_and_clean(record: Optional[Dict], deterministic: Dict, raw_text: str) -> Optional[Dict]:
    if not record:
        return None

    merged = {
        "name": deterministic.get("name") or record.get("name", ""),
        "email": deterministic.get("email") or record.get("email", ""),
        "phone": deterministic.get("phone") or record.get("phone", ""),
        "skills": record.get("skills", []),
        "years_experience": record.get("years_experience", 0),
        "summary": record.get("summary", ""),
        "work_experience": record.get("work_experience", []),
        "education": record.get("education", []),
        "certifications": record.get("certifications", []),
        "location": record.get("location"),
        "remote_available": record.get("remote_available"),
        "education_level": record.get("education_level"),
        "seniority": record.get("seniority"),
        "company_names": record.get("company_names", []),
        "job_titles": record.get("job_titles", []),
        "industry": record.get("industry"),
    }

    try:
        cv = CVSchema(**merged)
        confidence = cv.compute_confidence()
    except Exception:
        merged = repair_record(merged, raw_text, deterministic)
        try:
            cv = CVSchema(**merged)
            confidence = cv.compute_confidence()
        except Exception:
            return None

    name_lower = (cv.name or "").lower().strip()
    KNOWN_SKILLS_SET = {"python", "javascript", "react", "java", "docker"} # shortened for brevity
    if not cv.name or name_lower in KNOWN_SKILLS_SET:
        text_for_hash = f"{cv.email}{cv.skills}{cv.summary}".encode()
        placeholder_id = hashlib.sha256(text_for_hash).hexdigest()[:4].upper()
        cv.name = f"Candidate-{placeholder_id}"

    if len(cv.skills) < 1:
        return None

    industry = detect_industry(raw_text)
    text_for_hash = f"{cv.name}{cv.email}{cv.skills}{cv.summary}".encode()
    doc_hash = hashlib.sha256(text_for_hash).hexdigest()

    result = cv.dict()
    result.update({
        "_confidence": confidence,
        "_hash": doc_hash,
        "_raw_text": raw_text[:1000],
        "industry": industry,
    })
    return result

# ----------------------------------------------------------------------
# Main extraction function
# ----------------------------------------------------------------------
def extract_structured_data(pdf_path: str) -> List[Dict[str, Any]]:
    page_texts = extract_cvs_from_pdf(pdf_path)
    logger.info(f"Found {len(page_texts)} page(s) in {pdf_path}")

    client = genai.Client(api_key=config.require_google_api_key())
    model_name = get_working_model(client)
    logger.info(f"Using model: {model_name}")

    results = []
    for idx, text in enumerate(page_texts):
        if len(text) < config.EXTRACTION_MIN_TEXT_LENGTH:
            continue

        deterministic = {
            "email": extract_email(text),
            "phone": extract_phone(text),
            "name": extract_name(text),
        }

        try:
            raw = _call_gemini(client, model_name, text)
            cleaned = validate_and_clean(raw, deterministic, text)
            if cleaned:
                results.append(cleaned)
        except Exception as e:
            logger.error(f"Page {idx+1} failed: {e}")

    return results