"""
Central configuration for the AI Skill Intelligence Platform.
All thresholds, model names, and constants are defined here.
"""

# ─── Model Configuration ─────────────────────────────────────────────────────
SBERT_MODEL = "all-MiniLM-L6-v2"

# ─── Extraction Thresholds ────────────────────────────────────────────────────
FUZZY_THRESHOLD = 0.85          # token_set_ratio threshold for fuzzy matching
SEMANTIC_THRESHOLD = 0.75       # cosine similarity threshold for SBERT
KNN_NEIGHBORS = 10              # k for collaborative filtering

# ─── Skill Strength Weights (across layers) ──────────────────────────────────
LAYER_WEIGHTS = {
    "regex": 1.0,               # Exact match = max confidence
    "fuzzy": 0.85,              # Fuzzy match confidence
    "semantic": 0.80,           # Semantic similarity confidence
}

# ─── Gap Analysis ─────────────────────────────────────────────────────────────
GAP_SCORE_FORMULA = "market_demand * salary_impact / (difficulty * learning_hours)"

# ─── Roadmap Tier Thresholds ──────────────────────────────────────────────────
ROADMAP_TIERS = {
    "high_roi_low_diff": {"min_roi": 0.7, "max_difficulty": 2},
    "medium_roi_mod_diff": {"min_roi": 0.4, "max_difficulty": 4},
    "low_roi_necessary": {"min_roi": 0.0, "max_difficulty": 5},
}

# ─── Course Recommendation ────────────────────────────────────────────────────
MAX_COURSES_PER_SKILL = 5
MIN_COURSES_PER_SKILL = 3

# ─── File Paths ───────────────────────────────────────────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAXONOMY_PATH = os.path.join(BASE_DIR, "data", "skill_taxonomy.json")
COURSE_CATALOG_PATH = os.path.join(BASE_DIR, "data", "course_catalog.json")
EMBEDDING_CACHE_PATH = os.path.join(BASE_DIR, "data", "taxonomy_embeddings.npy")
COURSE_EMBEDDING_CACHE = os.path.join(BASE_DIR, "data", "course_embeddings.npy")

# ─── OCR Configuration ────────────────────────────────────────────────────────
OCR_DPI = 300
OCR_LANG = "eng"

# ─── Section Keywords ─────────────────────────────────────────────────────────
SECTION_KEYWORDS = {
    "education": [
        "education", "academic background", "qualifications",
        "degree", "university", "college", "school", "bachelor",
        "master", "phd", "certification", "coursework"
    ],
    "experience": [
        "experience", "work history", "employment", "career",
        "professional background", "job history", "positions held",
        "work experience", "professional experience", "employment history"
    ],
    "skills": [
        "skills", "technical skills", "competencies", "technologies",
        "expertise", "proficiencies", "tools", "programming languages",
        "frameworks", "soft skills", "core competencies", "skill set"
    ],
    "certifications": [
        "certifications", "certificates", "licenses", "awards",
        "achievements", "accreditations", "professional certifications",
        "credentials", "badges"
    ]
}
