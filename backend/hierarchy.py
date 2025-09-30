from urllib.parse import urlparse

# define our tiers
TIER = {
    "academic_regulations": 1,
    "general_info": 2,
    "course_desc": 3,
    "program": 4,
}
TIER_NAMES = {v: k for k, v in TIER.items()}


def compute_tier(page_url: str, title: str = "") -> dict:
    """
    classify a page into one of our tiers, based on its url and title.
    """
    tier = TIER["general_info"]   # default assume general info
    is_program = False
    level = "graduate"
    section = title or ""

    path = (urlparse(page_url or "").path or "").lower()

    # tier 1: academic regulations
    if "/graduate/academic-regulations-degree-requirements/" in path:
        tier = TIER["academic_regulations"]

    # tier 2: general information
    elif "/graduate/general-information/" in path:
        tier = TIER["general_info"]

    # tier 3: course descriptions
    elif "/graduate/course-descriptions/" in path or "/search/?" in path:
        tier = TIER["course_desc"]

    # tier 4: programs of study
    elif "/graduate/programs/" in path or "/graduate/programs-study/" in path:
        tier = TIER["program"]
        is_program = True

    # fallback: use title if url is unclear
    else:
        t = (title or "").lower()
        if any(kw in t for kw in ["academic regulations", "academic standards", "graduate grading"]):
            tier = TIER["academic_regulations"]

    return {
        "tier": tier,
        "tier_name": TIER_NAMES[tier],
        "is_program_page": is_program,
        "level": level,
        "section": section,
    }