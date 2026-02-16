# AI Skill Intelligence Platform

**Production-ready Python system for resume-to-JD skill gap analysis, learning roadmaps, and course recommendations.**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Descriptions](#module-descriptions)
3. [Folder Structure](#folder-structure)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Pipeline Explanation](#pipeline-explanation)
8. [Output Schema](#output-schema)

---

## Architecture Overview

```
Resume (PDF/DOCX)          Job Description (Text)
        │                          │
        ▼                          ▼
┌─────────────────┐    ┌─────────────────────────┐
│ A. Text         │    │ D. JD Skill Mining       │
│ Extraction      │    │ - Regex section detect   │
│ Engine          │    │ - Domain keyword scan    │
│ - PyPDF2/pypdf  │    │ - SBERT semantic search  │
│ - python-docx   │    │ - Industry heuristics    │
│ - OCR Fallback  │    └────────────┬────────────┘
│ - Normalization │                 │
│ - Section ID    │          Required Skills
└────────┬────────┘                 │
         │                          │
    Resume Text                     │
         │                          │
         ▼                          │
┌─────────────────┐                 │
│ B. Hybrid Skill │                 │
│ Extraction      │                 │
│ Layer 1: Regex  │                 │
│ Layer 2: Fuzzy  │                 │
│ Layer 3: SBERT  │                 │
└────────┬────────┘                 │
         │                          │
    Resume Skills                   │
         │                          │
         └────────────┬─────────────┘
                      │
                      ▼
         ┌─────────────────────────┐
         │ C. Skill Taxonomy       │
         │ - Canonical names       │
         │ - Aliases + ESCO/O*NET  │
         │ - Market demand scores  │
         │ - Difficulty levels     │
         │ - Salary impact scores  │
         └────────────┬────────────┘
                      │ Enrichment
                      ▼
         ┌─────────────────────────┐
         │ E. Skill Gap Analysis   │
         │ Gap Score formula:      │
         │ (Demand × Salary) /     │
         │ (Difficulty × Hours)    │
         │ + Topo-sort sequence    │
         └────────────┬────────────┘
                      │
              ┌───────┴────────┐
              │                │
              ▼                ▼
┌─────────────────┐   ┌─────────────────────┐
│ F. Learning     │   │ G. Course           │
│ Roadmap         │   │ Recommendation      │
│ - Tier 1/2/3    │   │ - TF-IDF (0.35)     │
│ - Timeline      │   │ - KNN k=10 (0.25)   │
│ - Milestones    │   │ - SBERT (0.40)      │
│ - Quick wins    │   │ Top 3-5 per skill   │
└─────────────────┘   └─────────────────────┘
```

---

## Module Descriptions

### A. Text Extraction Engine (`engines/text_extraction_engine.py`)

Multi-path extraction pipeline:
- **Digital PDF extraction**: Uses `pypdf` (PyPDF2 fallback)
- **DOCX extraction**: Uses `python-docx` to read paragraphs and table cells
- **OCR Fallback**: Triggered when digital extraction yields < 100 characters. Uses `pytesseract` + `pdf2image` (Poppler backend). Note: accuracy varies with scan quality.
- **Text Normalization**: Unicode normalization → special chars removal → whitespace normalization → lowercase standardization
- **Section Identification**: Rule-based keyword detection for Education, Experience, Skills, Certifications. Two-pass: header detection + fallback keyword search.

### B. Hybrid Skill Extraction Engine (`engines/hybrid_skill_extraction_engine.py`)

Three-layer pipeline:

| Layer | Method | Threshold | Precision | Recall |
|-------|--------|-----------|-----------|--------|
| 1 | Regex word-boundary matching | Exact | Very High | Low |
| 2 | Fuzzy `token_set_ratio` + alias map | 0.85 | High | Medium |
| 3 | SBERT cosine similarity | 0.75 | Medium | High |

**Layer 3 Example**: `"optimized distributed data pipelines"` → detects **Apache Spark**, **Distributed Systems**

Strength score aggregated across layers: `regex=1.0`, `fuzzy=score×0.85`, `semantic=score×0.80`

### C. Skill Taxonomy System (`engines/skill_taxonomy_system.py`)

- 30 pre-built skills (ESCO/O*NET-inspired), easily extensible
- O(1) lookup by canonical name or any alias
- Dynamic CRUD: `add_skill()`, `update_skill()`, `delete_skill()`
- Auto-persists to JSON + invalidates embedding cache
- Semantic search: find related skills via SBERT similarity

### D. JD Skill Mining (`engines/jd_skill_miner.py`)

Optimized "JD Mode" (fewer semantic passes than resume mode):
1. Industry detection from domain keywords (finance, tech, healthcare, e-commerce)
2. "Required Skills" section regex isolation (6 pattern variants)
3. Regex exact matching
4. Fuzzy keyword scan (1-gram and 2-gram only in JD mode)
5. SBERT semantic matching (every 2nd sentence, max 20 sentences)
6. Industry heuristic boosting (e.g., Java gets 1.3× in finance contexts)

### E. Skill Gap Analysis Engine (`engines/skill_gap_analysis_engine.py`)

For each missing skill:
```
Gap Score = (Market Demand × Salary Impact) / (Normalized Difficulty × Normalized Hours)
```

**Acquisition sequence**: Kahn's topological sort respects prerequisite dependencies (e.g., Python must come before Django).

### F. Learning Roadmap Generator (`engines/learning_roadmap_generator.py`)

3-tier structure:
- **Tier 1**: gap_score ≥ 0.7 AND difficulty ≤ 2 → Quick wins
- **Tier 2**: gap_score ≥ 0.4 AND difficulty ≤ 4 → Core competency
- **Tier 3**: Everything else → Necessary but lower ROI

Each skill entry includes: weeks to proficiency, learning type, prerequisites, proficiency milestone, study schedule.

### G. Course Recommendation Engine (`engines/course_recommendation_engine.py`)

Hybrid fusion with weights:
- **Content-Based (TF-IDF)**: 0.35 weight — TF-IDF cosine similarity with bigram tokenization
- **Collaborative (KNN, k=10)**: 0.25 weight — 200-user synthetic interaction matrix
- **Semantic (SBERT)**: 0.40 weight — sentence embedding similarity

Returns top 3-5 courses per gap skill with semantic score and rationale.

---

## Folder Structure

```
skill_intelligence_platform/
├── main.py                          # Entry point + demo
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── config/
│   ├── __init__.py
│   └── settings.py                  # All thresholds, model names, paths
├── engines/
│   ├── __init__.py
│   ├── text_extraction_engine.py    # Module A
│   ├── hybrid_skill_extraction_engine.py  # Module B
│   ├── skill_taxonomy_system.py     # Module C
│   ├── jd_skill_miner.py            # Module D
│   ├── skill_gap_analysis_engine.py # Module E
│   ├── learning_roadmap_generator.py # Module F
│   └── course_recommendation_engine.py  # Module G
├── utils/
│   ├── __init__.py
│   ├── embedding_service.py         # Shared SBERT singleton + caching
│   └── compat.py                    # Library compatibility shims
├── data/
│   ├── skill_taxonomy.json          # 30-skill taxonomy (ESCO/O*NET-inspired)
│   └── course_catalog.json          # 20-course catalog
├── examples/
│   ├── example_resume.txt           # Sample Data Engineer resume
│   └── example_jd.txt               # Sample ML Engineer JD
└── outputs/
    └── analysis_result.json         # Generated output
```

---

## Installation

```bash
# Install required packages
pip install PyPDF2 python-docx pytesseract pdf2image sentence-transformers scikit-learn numpy pandas rapidfuzz scipy

# For OCR support (Poppler)
# Ubuntu/Debian:
sudo apt-get install poppler-utils tesseract-ocr

# macOS:
brew install poppler tesseract
```

---

## Usage

### Full Pipeline Analysis

```python
from main import SkillIntelligencePlatform

platform = SkillIntelligencePlatform(weekly_hours=15)

# From file path
result = platform.analyze(
    resume_path="path/to/resume.pdf",
    jd_text=open("path/to/jd.txt").read(),
    weekly_hours=15
)

# From raw text
result = platform.analyze(
    resume_text="John Smith... Python... SQL...",
    jd_text="We need: Python, Machine Learning, PyTorch..."
)

# Save results
platform.save_result(result, "outputs/my_analysis.json")
```

### Individual Module Usage

```python
# Text Extraction only
from engines.text_extraction_engine import TextExtractionEngine
engine = TextExtractionEngine()
sections = engine.extract("resume.pdf")
# → {"education": "...", "experience": "...", "skills": "...", ...}

# Skill Extraction only
from engines.hybrid_skill_extraction_engine import HybridSkillExtractionEngine
extractor = HybridSkillExtractionEngine()
skills = extractor.extract_skills(text="Python ML engineer...", mode="resume")
# → [{"skill_name": "Python", "strength_score": 1.0, "extraction_method": "regex", ...}]

# JD Mining only
from engines.jd_skill_miner import JobDescriptionSkillMiner
miner = JobDescriptionSkillMiner()
result = miner.mine_skills(jd_text="Required: Python, ML, Docker...")
# → {"required_skills": [...], "detected_industry": "tech", ...}

# Taxonomy operations
from engines.skill_taxonomy_system import SkillTaxonomySystem
taxonomy = SkillTaxonomySystem()
skill = taxonomy.get_skill("Machine Learning")
taxonomy.add_skill({"canonical_name": "LangChain", "category": "Technical", ...})
related = taxonomy.semantic_search("natural language generation", top_k=5)
```

---

## Configuration

All thresholds are centralized in `config/settings.py`:

| Setting | Value | Description |
|---------|-------|-------------|
| `FUZZY_THRESHOLD` | 0.85 | Minimum token_set_ratio for fuzzy matching |
| `SEMANTIC_THRESHOLD` | 0.75 | Minimum cosine similarity for SBERT matching |
| `KNN_NEIGHBORS` | 10 | k for collaborative filtering |
| `SBERT_MODEL` | `all-MiniLM-L6-v2` | Sentence-BERT model name |
| `MAX_COURSES_PER_SKILL` | 5 | Max course recommendations per skill |

---

## Output Schema

```json
{
  "metadata": {
    "platform": "AI Skill Intelligence Platform v1.0",
    "processing_time_seconds": 5.9,
    "weekly_hours_assumed": 15
  },
  "resume_analysis": {
    "sections_detected": {"education": true, "experience": true, "skills": true, "certifications": true},
    "skills_found": 14,
    "top_skills": [
      {
        "skill_name": "Python",
        "normalized_name": "python",
        "strength_score": 1.0,
        "extraction_method": "regex",
        "type": "Technical"
      }
    ]
  },
  "jd_analysis": {
    "skills_required": 13,
    "detected_industry": "tech",
    "required_skills": [...]
  },
  "gap_analysis": {
    "summary": {
      "total_required": 13,
      "total_matched": 6,
      "total_gaps": 7,
      "match_percentage": 46.2,
      "weighted_avg_severity": 2.76,
      "profile_strength": "Needs Work"
    },
    "matched_skills": [...],
    "skill_gaps": [
      {
        "skill_name": "CI/CD",
        "gap_score": 6.02,
        "difficulty": 3,
        "learning_hours": 60,
        "market_demand": 0.85,
        "salary_impact": 0.80
      }
    ],
    "acquisition_sequence": [
      {"step": 1, "skill_name": "CI/CD", "reason": "High ROI..."},
      {"step": 2, "skill_name": "PyTorch", ...}
    ]
  },
  "roadmap": {
    "roadmap_summary": {"total_weeks_estimated": 66, "total_hours_required": 990},
    "quick_wins": [...],
    "tiers": {
      "tier_1": {"name": "High ROI + Low Difficulty", "skills": [...]},
      "tier_2": {"name": "Medium ROI + Moderate Difficulty", "skills": [...]},
      "tier_3": {"name": "Low ROI but Necessary", "skills": [...]}
    },
    "milestones": [...]
  },
  "course_recommendations": {
    "PyTorch": [
      {
        "course_name": "PyTorch for Deep Learning Bootcamp",
        "platform": "Udemy",
        "level": "Intermediate",
        "duration_hours": 30,
        "semantic_score": 0.418,
        "hybrid_score": 0.312,
        "rationale": "Recommended because: covers relevant keywords..."
      }
    ]
  }
}
```
