"""
Pydantic models for CV data validation and confidence scoring.
Includes structured work experience, education level, seniority, location, remote availability,
industry domain, tech cluster, and helper functions for date parsing and classification.
"""
from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field, validator, EmailStr
import re
from datetime import datetime

class WorkExperience(BaseModel):
    """Single work experience entry."""
    job_title: str = ""
    company: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""
    duration_months: Optional[int] = None

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse a date string into a datetime object."""
    if not date_str or not isinstance(date_str, str):
        return None
    date_str = date_str.strip()
    if date_str.lower() == "present":
        return datetime.now()
    # Try common formats
    for fmt in ("%b %Y", "%B %Y", "%Y-%m", "%m/%Y", "%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def compute_experience_from_work(work_list: List[Dict]) -> float:
    """Compute total years of experience from a list of work experience dicts."""
    total_months = 0
    for w in work_list:
        start = parse_date(w.get("start_date"))
        end = parse_date(w.get("end_date"))
        if start:
            if not end or (isinstance(end, datetime) and end > start):
                end = datetime.now()
            delta = end - start
            total_months += delta.days / 30.44  # approximate
    return round(total_months / 12, 2)

def classify_education_level(education_texts: List[str]) -> Optional[str]:
    """
    Determine highest education level from a list of education strings.
    Returns 'phd', 'master', 'bachelor', or None.
    """
    level_keywords = {
        'phd': ['phd', 'doctor of philosophy', 'doctorate', 'd.phil', 'd.sc'],
        'master': ['master', 'msc', 'm.sc', 'ma', 'm.a', 'mba', 'm.tech', 'mtech', 'm.e', 'meng'],
        'bachelor': ['bachelor', 'b.sc', 'bsc', 'ba', 'b.a', 'b.tech', 'btech', 'b.e', 'beng', 'bcs', 'b.cs']
    }
    highest = None
    text = ' '.join(education_texts).lower()
    for level, keywords in level_keywords.items():
        for kw in keywords:
            if kw in text:
                if level == 'phd':
                    return 'phd'
                if level == 'master' and highest != 'phd':
                    highest = 'master'
                if level == 'bachelor' and highest not in ('phd', 'master'):
                    highest = 'bachelor'
    return highest

def classify_seniority(job_titles: List[str], summary: str = "") -> Optional[str]:
    """
    Infer seniority level from job titles and summary.
    Returns 'junior', 'mid', 'senior', 'lead', 'principal', or None.
    """
    text = ' '.join(job_titles).lower() + ' ' + summary.lower()
    if any(kw in text for kw in ['principal', 'staff', 'architect']):
        return 'principal'
    if any(kw in text for kw in ['lead', 'head', 'manager', 'director']):
        return 'lead'
    if any(kw in text for kw in ['senior', 'sr']):
        return 'senior'
    if any(kw in text for kw in ['junior', 'jr', 'entry']):
        return 'junior'
    if any(kw in text for kw in ['mid', 'intermediate']):
        return 'mid'
    return None

class CVSchema(BaseModel):
    """Strict schema for extracted CV data."""
    name: str = Field(..., min_length=1)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    summary: str = Field("", max_length=2000)

    work_experience: List[WorkExperience] = Field(default_factory=list)

    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)

    # Computed fields
    education_level: Optional[Literal['bachelor', 'master', 'phd']] = None
    seniority: Optional[Literal['junior', 'mid', 'senior', 'lead', 'principal']] = None
    company_names: List[str] = Field(default_factory=list)
    job_titles: List[str] = Field(default_factory=list)

    location: Optional[str] = None
    remote_available: Optional[bool] = None

    years_experience: float = Field(0.0, ge=0)

    # Data intelligence fields
    industry: Optional[str] = None
    tech_cluster: Optional[int] = None

    # Internal fields (not part of validation, but will be present in final dict)
    _raw_text: Optional[str] = None
    _confidence: float = 0.0
    _hash: str = ""

    @validator('skills', pre=True, always=True)
    def validate_skills(cls, v):
        """Convert comma-separated string to list if needed."""
        if isinstance(v, str):
            v = re.split(r'[,\n;|]+', v)
        if isinstance(v, list):
            return [s.strip() for s in v if s and s.strip()]
        return []

    @validator('phone', pre=True, always=True)
    def clean_phone(cls, v):
        """Remove non-numeric characters except + and spaces."""
        if v and isinstance(v, str):
            return re.sub(r'[^\d+\s]', '', v).strip()
        return v

    @validator('years_experience', pre=True, always=True)
    def compute_years_experience(cls, v, values):
        """If years_experience is not provided or zero, compute from work_experience."""
        if v and v > 0:
            return v
        work_list = values.get('work_experience', [])
        if work_list:
            return compute_experience_from_work([w.dict() if isinstance(w, WorkExperience) else w for w in work_list])
        return 0.0

    @validator('education_level', pre=True, always=True)
    def compute_education_level(cls, v, values):
        """If education_level is missing, classify from education strings."""
        if v:
            return v
        edu_list = values.get('education', [])
        if edu_list:
            return classify_education_level(edu_list)
        return None

    @validator('seniority', pre=True, always=True)
    def compute_seniority(cls, v, values):
        """If seniority is missing, infer from job titles and summary."""
        if v:
            return v
        job_titles = values.get('job_titles', [])
        summary = values.get('summary', '')
        return classify_seniority(job_titles, summary)

    @validator('company_names', pre=True, always=True)
    def compute_company_names(cls, v, values):
        """Aggregate unique company names from work_experience."""
        if v:
            return v
        work_list = values.get('work_experience', [])
        companies = []
        for w in work_list:
            if isinstance(w, dict):
                company = w.get('company')
            else:
                company = getattr(w, 'company', None)
            if company and company.strip():
                companies.append(company.strip())
        return list(dict.fromkeys(companies))  # unique

    @validator('job_titles', pre=True, always=True)
    def compute_job_titles(cls, v, values):
        """Aggregate unique job titles from work_experience."""
        if v:
            return v
        work_list = values.get('work_experience', [])
        titles = []
        for w in work_list:
            if isinstance(w, dict):
                title = w.get('job_title')
            else:
                title = getattr(w, 'job_title', None)
            if title and title.strip():
                titles.append(title.strip())
        return list(dict.fromkeys(titles))

    def compute_confidence(self) -> float:
        """
        Compute a confidence score based on field completeness.
        Returns a float between 0 and 1.
        """
        score = 0.0
        total_weight = 0.0

        fields_weights = {
            'name': 2.0,
            'email': 1.5,
            'phone': 1.0,
            'skills': 2.0,
            'years_experience': 1.5,
            'summary': 1.0,
            'work_experience': 2.0,
            'education': 1.0,
            'education_level': 1.0,
            'seniority': 1.0,
            'company_names': 0.5,
            'job_titles': 0.5,
            'certifications': 0.5,
            'location': 0.5,
            'remote_available': 0.5,
            'industry': 0.5,       # added
            'tech_cluster': 0.5,    # added
        }

        for field, weight in fields_weights.items():
            total_weight += weight
            value = getattr(self, field, None)
            if field == 'skills':
                if value and len(value) >= 2:
                    score += weight
                elif value and len(value) == 1:
                    score += weight * 0.5
            elif field == 'work_experience':
                if value and len(value) > 0:
                    score += weight
            elif field == 'years_experience':
                if value and value > 0:
                    score += weight
                else:
                    score += weight * 0.3
            elif field in ('education_level', 'seniority', 'industry'):
                if value:
                    score += weight
            elif field == 'tech_cluster':
                if value is not None:
                    score += weight
            elif field in ('education', 'certifications', 'company_names', 'job_titles', 'location', 'remote_available'):
                if value and (isinstance(value, bool) or str(value).strip()):
                    score += weight
            else:
                if value and str(value).strip():
                    score += weight

        if self.email and self.phone:
            score += 0.5
            total_weight += 0.5

        self._confidence = round(score / max(total_weight, 1.0), 3)
        return self._confidence

    class Config:
        extra = 'forbid'