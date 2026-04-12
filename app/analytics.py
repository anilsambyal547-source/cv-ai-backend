"""
Data Intelligence: seniority, tech clusters, industry detection, skill vectors.
"""
from typing import List, Dict, Any
from collections import Counter


# ── Seniority ──────────────────────────────────────────────────────────────────

SENIORITY_RULES = [
    (lambda y, t: y >= 10 or any(x in t for x in ["principal", "staff", "distinguished", "fellow"]), "principal"),
    (lambda y, t: y >= 7  or any(x in t for x in ["lead", "architect", "head of", "director"]),      "lead"),
    (lambda y, t: y >= 4  or any(x in t for x in ["senior", "sr.", "sr "]),                           "senior"),
    (lambda y, t: y >= 2  or any(x in t for x in ["mid", "intermediate", "associate"]),               "mid"),
    (lambda y, t: True,                                                                                "junior"),
]

def compute_seniority(years_experience: float, title: str = "") -> str:
    title_lower = title.lower()
    for rule, level in SENIORITY_RULES:
        if rule(years_experience, title_lower):
            return level
    return "junior"


# ── Tech Stack Clusters ────────────────────────────────────────────────────────

TECH_CLUSTERS: Dict[str, List[str]] = {
    "frontend":     ["react", "vue", "angular", "nextjs", "svelte", "typescript", "javascript", "css", "tailwind", "html", "webpack"],
    "backend":      ["python", "node", "fastapi", "django", "flask", "express", "java", "spring", "go", "rust", "php", "laravel", "rails"],
    "data":         ["pandas", "numpy", "spark", "kafka", "airflow", "dbt", "sql", "postgresql", "mysql", "mongodb", "elasticsearch"],
    "ml_ai":        ["tensorflow", "pytorch", "scikit", "huggingface", "langchain", "openai", "keras", "mlflow", "xgboost", "llm"],
    "devops":       ["docker", "kubernetes", "aws", "gcp", "azure", "terraform", "jenkins", "github actions", "ansible", "ci/cd"],
    "mobile":       ["react native", "flutter", "swift", "kotlin", "ios", "android", "expo"],
    "security":     ["penetration", "owasp", "siem", "splunk", "vault", "iam", "soc", "cybersecurity"],
}

def compute_tech_clusters(skills: List[str]) -> Dict[str, float]:
    """Returns cluster → confidence score (0–1)."""
    skills_lower = [s.lower() for s in skills]
    scores: Dict[str, float] = {}
    for cluster, keywords in TECH_CLUSTERS.items():
        matches = sum(1 for k in keywords if any(k in s for s in skills_lower))
        if matches > 0:
            scores[cluster] = round(min(matches / max(len(keywords) * 0.4, 1), 1.0), 2)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ── Industry Detection ─────────────────────────────────────────────────────────

INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "fintech":       ["banking", "payment", "finance", "trading", "blockchain", "defi", "crypto", "insurance", "lending", "kyc"],
    "healthcare":    ["ehr", "clinical", "hospital", "medical", "pharma", "health", "dicom", "fhir", "patient", "drug"],
    "ecommerce":     ["ecommerce", "shopify", "retail", "marketplace", "inventory", "cart", "checkout", "fulfillment"],
    "saas":          ["b2b", "multi-tenant", "subscription", "mrr", "arr", "churn", "onboarding", "saas", "crm"],
    "gaming":        ["unity", "unreal", "game", "graphics", "opengl", "directx", "shader", "multiplayer"],
    "cybersecurity": ["security", "penetration", "vulnerability", "soc", "threat", "incident", "malware", "compliance"],
    "edtech":        ["lms", "elearning", "education", "course", "curriculum", "student", "teacher"],
    "logistics":     ["supply chain", "logistics", "fleet", "route", "warehouse", "shipment", "tracking"],
}

def detect_industry(text: str, skills: List[str]) -> str:
    combined = (text + " " + " ".join(skills)).lower()
    scores   = {
        industry: sum(1 for k in keywords if k in combined)
        for industry, keywords in INDUSTRY_KEYWORDS.items()
    }
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "general"


# ── Skill Vector ───────────────────────────────────────────────────────────────

ALL_SKILLS = sorted({k for v in TECH_CLUSTERS.values() for k in v})

def compute_skill_vector(skills: List[str]) -> List[float]:
    """Binary skill presence vector over ALL_SKILLS vocabulary."""
    skills_lower = [s.lower() for s in skills]
    return [
        1.0 if any(vocab in s for s in skills_lower) else 0.0
        for vocab in ALL_SKILLS
    ]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot   = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a ** 2 for a in v1) ** 0.5
    norm2 = sum(b ** 2 for b in v2) ** 0.5
    return round(dot / (norm1 * norm2), 4) if norm1 and norm2 else 0.0


# ── Master Enrichment ─────────────────────────────────────────────────────────

def enrich_candidate(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich a candidate metadata dict with all intelligence fields."""
    skills_raw = metadata.get("skills", "") or ""
    skills     = [s.strip() for s in skills_raw.split(",") if s.strip()]
    years      = float(metadata.get("years_experience", 0) or 0)
    title      = metadata.get("title", "") or ""
    bio        = metadata.get("document", "") or ""

    metadata["seniority"]     = compute_seniority(years, title)
    metadata["tech_clusters"] = compute_tech_clusters(skills)
    metadata["industry"]      = detect_industry(bio, skills)
    metadata["skill_vector"]  = compute_skill_vector(skills)
    return metadata