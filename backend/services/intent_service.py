import re
from typing import Optional, Dict, List, Any, Tuple
from utils.course_utils import detect_course_code, COURSE_CODE_RX

SECTION_STOPWORDS = tuple([
    "upon completion", "program learning outcomes", "admission requirements",
    "requirements", "application requirements", "core courses", "electives",
    "overview", "policies", "sample", "plan of study"
])

# --- Policy term detector
def has_policy_terms(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in [
        "gpa", "good standing", "probation", "dismissal", "grade", "b-", "b minus", "c grade", "c-", "c minus", "minimum gpa",
        "committee", "guidance committee", "supervisory committee", "qualifying exam", "qualifying examination",
        "final exam", "exam attempt", "examination attempt"
    ])

# --- Admissions term detector and URL helpers ---
def has_admissions_terms(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in [
        "admission", "admissions", "apply", "application", "requirements", "gre", "gmat",
        "test score", "test scores", "english proficiency", "toefl", "ielts",
        "letters of recommendation", "recommendation letter"
    ])

def is_admissions_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/general-information/admissions/" in url

def is_degree_requirements_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/academic-regulations-degree-requirements/degree-requirements/" in url
