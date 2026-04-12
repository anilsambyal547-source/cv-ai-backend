"""
Job description parser – extracts skills, experience, and education from JD text.
"""
import re
from typing import Set, Optional, Dict, List
import logging

from config import config

logger = logging.getLogger(__name__)

def extract_skills_from_jd(text: str) -> Set[str]:
    """Extract canonical skills from JD using synonym dictionary."""
    text_lower = text.lower()
    found = set()
    for canon, variants in config.SKILL_SYNONYMS.items():
        if canon in text_lower:
            found.add(canon)
        else:
            for var in variants:
                if var in text_lower:
                    found.add(canon)
                    break
    return found

def extract_experience_from_jd(text: str) -> Optional[float]:
    """
    Extract minimum years of experience from JD.
    Looks for patterns like "5+ years", "3-5 years", "at least 3 years".
    Returns the minimum required years as float, or None.
    """
    patterns = [
        r'(\d+)[\+]?\s*(?:years?|yrs?)',
        r'(\d+)[-\s]to[-\s](\d+)\s*(?:years?|yrs?)',
        r'(\d+)[-\s](\d+)\s*(?:years?|yrs?)',
        r'at\s*least\s*(\d+)\s*(?:years?|yrs?)',
        r'minimum\s*(\d+)\s*(?:years?|yrs?)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            if len(match.groups()) == 1:
                return float(match.group(1))
            elif len(match.groups()) == 2:
                # return the lower bound
                return float(match.group(1))
    return None

def extract_education_from_jd(text: str) -> Optional[str]:
    """
    Extract required education level from JD.
    Returns 'phd', 'master', 'bachelor', or None.
    """
    level_keywords = {
        'phd': ['phd', 'doctorate', 'doctor of philosophy'],
        'master': ['master', 'msc', 'm.sc', 'ma', 'm.a', 'mba', 'graduate degree'],
        'bachelor': ['bachelor', 'b.sc', 'bsc', 'ba', 'b.a', 'undergraduate degree']
    }
    text_lower = text.lower()
    for level, keywords in level_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return level
    return None

def parse_job_description(text: str) -> Dict:
    """
    Parse job description and return structured requirements.
    Returns dict with keys: skills (set), min_experience (float or None), education (str or None).
    """
    skills = extract_skills_from_jd(text)
    min_exp = extract_experience_from_jd(text)
    education = extract_education_from_jd(text)
    logger.info(f"JD parsed: {len(skills)} skills, experience={min_exp}, education={education}")
    return {
        "skills": skills,
        "min_experience": min_exp,
        "education": education,
    }