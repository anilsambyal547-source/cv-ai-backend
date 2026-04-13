"""
AI‑powered candidate insights: summary, strengths, weaknesses, interview questions.
Uses Gemini with structured prompts, enforces JSON output, and robust parsing.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional

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

logger = logging.getLogger(__name__)

_WORKING_MODEL = None

def get_working_model(client) -> str:
    global _WORKING_MODEL
    if _WORKING_MODEL is not None:
        return _WORKING_MODEL

    if config.PREFERRED_MODEL:
        try:
            client.models.get(model=config.PREFERRED_MODEL)
            logger.info(f"[OK] Using preferred model: {config.PREFERRED_MODEL}")
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
                    logger.info(f"[OK] Using model: {model_id}")
                    _WORKING_MODEL = model_id
                    return _WORKING_MODEL
            except Exception:
                continue

        for m in models:
            name = m.name.replace("models/", "")
            if any(x in name for x in ["audio", "image", "vision", "tts", "er-"]):
                continue
            if "generateContent" in m.supported_actions:
                logger.info(f"[OK] Using fallback model: {name}")
                _WORKING_MODEL = name
                return _WORKING_MODEL
    except Exception as e:
        logger.error(f"Model discovery error: {e}")

    raise RuntimeError("[ERROR] No suitable Gemini model found. Check API key and quota.")

def _repair_json(raw: str) -> Optional[str]:
    """
    Attempt to extract a JSON object from the response.
    - Remove markdown fences
    - Find the first '{' and last '}'
    """
    raw = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.DOTALL)
    raw = raw.strip()
    start = raw.find('{')
    end = raw.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None
    return raw[start:end+1]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def _call_gemini(prompt: str) -> Optional[str]:
    """Call Gemini with the given prompt, request JSON output."""
    client = genai.Client(api_key=config.require_google_api_key())
    model_name = get_working_model(client)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=config.EXTRACTION_MAX_TOKENS,  # Should be 4096
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
        return text.strip()
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        raise

def _format_cv_data(cv: Dict[str, Any]) -> str:
    """Format CV data into a readable text for the prompt."""
    lines = []
    lines.append(f"Name: {cv.get('name', 'Unknown')}")
    if cv.get('email'):
        lines.append(f"Email: {cv['email']}")
    if cv.get('phone'):
        lines.append(f"Phone: {cv['phone']}")
    lines.append(f"Years of Experience: {cv.get('years_experience', 0)}")
    skills = cv.get('skills', [])
    if skills:
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(',') if s.strip()]
        lines.append(f"Skills: {', '.join(skills)}")
    if cv.get('summary'):
        lines.append(f"Summary: {cv['summary']}")
    
    work_exp = cv.get('work_experience')
    if work_exp:
        lines.append("Work Experience:")
        if isinstance(work_exp, str):
            lines.append(work_exp)
        elif isinstance(work_exp, list):
            for we in work_exp:
                if isinstance(we, dict):
                    lines.append(f"- {we.get('job_title', '')} at {we.get('company', '')} ({we.get('start_date', '')} - {we.get('end_date', '')})")
                else:
                    lines.append(f"- {we}")

    education = cv.get('education')
    if education:
        lines.append("Education:")
        if isinstance(education, str):
            lines.append(education)
        elif isinstance(education, list):
            for edu in education:
                lines.append(f"- {edu}")

    certifications = cv.get('certifications')
    if certifications:
        lines.append("Certifications:")
        if isinstance(certifications, str):
            lines.append(certifications)
        elif isinstance(certifications, list):
            for cert in certifications:
                lines.append(f"- {cert}")
    return "\n".join(lines)

def generate_insights(
    cv_data: Dict[str, Any],
    job_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate AI insights for a candidate.
    Returns dict with keys: summary, strengths, weaknesses, interview_questions.
    """
    cv_text = _format_cv_data(cv_data)

    if job_description:
        prompt = f"""
You are an expert technical recruiter. Given the following candidate profile and a job description, generate:

1. A concise professional summary (2‑3 sentences).
2. A list of key strengths (3‑5 bullet points).
3. A list of weaknesses or gaps relative to the job description (if any; be specific about missing skills or experience).
4. A list of 3‑5 insightful interview questions tailored to the candidate's background and the job requirements.

Format your response as JSON with keys: "summary", "strengths", "weaknesses", "interview_questions" (each as a list of strings). Output only the JSON, no additional text.

Candidate Profile:
{cv_text}

Job Description:
{job_description}
"""
    else:
        prompt = f"""
You are an expert technical recruiter. Given the following candidate profile, generate:

1. A concise professional summary (2‑3 sentences).
2. A list of key strengths (3‑5 bullet points).
3. A list of general weaknesses or areas for improvement (if any; if none, use an empty list).
4. A list of 3‑5 insightful interview questions to probe the candidate's experience.

Format your response as JSON with keys: "summary", "strengths", "weaknesses", "interview_questions" (each as a list of strings). Output only the JSON, no additional text.

Candidate Profile:
{cv_text}
"""

    response_text = _call_gemini(prompt)
    if not response_text:
        logger.error("Empty response from Gemini")
        return {
            "summary": "Error generating insights (empty response).",
            "strengths": [],
            "weaknesses": [],
            "interview_questions": []
        }

    # Attempt to repair and parse JSON
    repaired = _repair_json(response_text)
    if not repaired:
        logger.error("No JSON object found in response")
        return {
            "summary": "Error parsing insights (no JSON found).",
            "strengths": [],
            "weaknesses": [],
            "interview_questions": []
        }

    try:
        data = json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON even after repair: {e}")
        logger.error(f"Repaired JSON string: {repaired}")
        return {
            "summary": "Error parsing insights (invalid JSON).",
            "strengths": [],
            "weaknesses": [],
            "interview_questions": []
        }

    # Ensure expected keys exist
    result = {
        "summary": data.get("summary", ""),
        "strengths": data.get("strengths", []),
        "weaknesses": data.get("weaknesses", []),
        "interview_questions": data.get("interview_questions", []),
    }
    return result