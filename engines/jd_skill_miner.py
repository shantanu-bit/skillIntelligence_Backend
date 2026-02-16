"""
Module D: Job Description Skill Mining
========================================
Optimized "JD Mode" for extracting required skills from job descriptions.

Faster and less exhaustive than resume mode:
1. Regex detection of "Required Skills" sections
2. Domain keyword scanning
3. Semantic extraction using SBERT (all-MiniLM-L6-v2)
4. Industry heuristics (e.g., prioritize Java in finance)

Fewer semantic passes vs Resume mode.

Output: List of required skills with relevance scores.
"""

import re
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Set

from utils.embedding_service import get_embedding_service
from config.settings import FUZZY_THRESHOLD, SEMANTIC_THRESHOLD, TAXONOMY_PATH
from utils.compat import fuzz

logger = logging.getLogger(__name__)


# ─── Industry Heuristic Rules ─────────────────────────────────────────────────
# Boost weights for certain skills in specific industries
INDUSTRY_HEURISTICS = {
    "finance": {
        "Java": 1.3,
        "Python": 1.2,
        "SQL": 1.3,
        "Data Science": 1.2,
        "Machine Learning": 1.2,
        "Statistics": 1.3,
        "Risk Management": 1.4,
    },
    "tech": {
        "Python": 1.2,
        "Machine Learning": 1.2,
        "Docker": 1.2,
        "Kubernetes": 1.2,
        "AWS": 1.2,
        "React": 1.1,
    },
    "healthcare": {
        "Python": 1.1,
        "SQL": 1.2,
        "Machine Learning": 1.1,
        "Data Science": 1.2,
        "Communication": 1.3,
    },
    "ecommerce": {
        "Python": 1.2,
        "React": 1.3,
        "SQL": 1.2,
        "Data Engineering": 1.2,
    }
}

# Regex patterns to find "Required Skills" sections in JDs
REQUIRED_SECTION_PATTERNS = [
    r"(?i)required\s+skills?[\s\:\-]+([\s\S]{0,800}?)(?=preferred|responsibilities|about|$)",
    r"(?i)must\s+have[\s\:\-]+([\s\S]{0,800}?)(?=nice|preferred|about|$)",
    r"(?i)technical\s+requirements?[\s\:\-]+([\s\S]{0,800}?)(?=preferred|responsibilities|about|$)",
    r"(?i)qualifications?[\s\:\-]+([\s\S]{0,600}?)(?=preferred|responsibilities|about|$)",
    r"(?i)what\s+we(?:'re|\s+are)\s+looking\s+for[\s\:\-]+([\s\S]{0,600}?)(?=what\s+we\s+offer|about|$)",
]

# Domain indicator terms → industry category
DOMAIN_INDICATORS = {
    "finance": ["bank", "financial", "trading", "fintech", "quant", "hedge fund", "portfolio", "equity"],
    "healthcare": ["hospital", "clinical", "medical", "pharma", "biotech", "ehr", "patient"],
    "ecommerce": ["e-commerce", "marketplace", "retail", "shopify", "magento", "woocommerce"],
    "tech": ["saas", "software", "platform", "developer", "engineering", "cloud", "startup"],
}


class JobDescriptionSkillMiner:
    """
    Extracts required skills from job descriptions using a 4-step pipeline:
    1. Regex-based required section extraction
    2. Domain keyword scanning
    3. Optimized SBERT semantic matching
    4. Industry heuristic scoring
    """

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.taxonomy = self._load_taxonomy()
        self._build_structures()
        logger.info("[JDMiner] Initialized.")

    def _load_taxonomy(self) -> List[Dict]:
        with open(TAXONOMY_PATH, "r") as f:
            return json.load(f)["skills"]

    def _build_structures(self):
        """Build lookup structures for regex and fuzzy scanning."""
        self.alias_map: Dict[str, str] = {}      # alias_lower -> canonical
        self.all_tokens: List[tuple] = []        # (token_lower, canonical)
        self.category_map: Dict[str, Dict] = {}  # canonical -> metadata

        for skill in self.taxonomy:
            canonical = skill["canonical_name"]
            self.category_map[canonical] = {
                "type": skill.get("category", "Technical"),
                "market_demand": skill.get("market_demand", 0.5),
                "salary_impact": skill.get("salary_impact", 0.5),
                "difficulty": skill.get("difficulty", 3),
                "learning_hours": skill.get("learning_hours", 80),
            }
            for name in [canonical] + skill.get("aliases", []):
                self.alias_map[name.lower()] = canonical
                self.all_tokens.append((name.lower(), canonical))

        # Compiled regex patterns for quick keyword scan
        self.skill_patterns = []
        for alias_lower, canonical in sorted(self.all_tokens, key=lambda x: len(x[0]), reverse=True):
            escaped = re.escape(alias_lower)
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
            self.skill_patterns.append((pattern, canonical))

    # ─── Main Interface ───────────────────────────────────────────────────────

    def mine_skills(self, jd_text: str) -> Dict:
        """
        Extract required skills from a job description.

        Args:
            jd_text: Raw job description text

        Returns:
            Dict with:
            - required_skills: List of skill dicts with relevance scores
            - detected_industry: Inferred industry
            - jd_summary: Brief analysis
        """
        if not jd_text.strip():
            return {"required_skills": [], "detected_industry": "unknown", "jd_summary": "Empty JD"}

        jd_lower = jd_text.lower()
        logger.info(f"[JDMiner] Mining JD ({len(jd_text)} chars)...")

        # Step 1: Detect industry for heuristic application
        industry = self._detect_industry(jd_lower)
        logger.info(f"[JDMiner] Detected industry: {industry}")

        # Step 2: Extract "Required Skills" section text if present
        required_section = self._extract_required_section(jd_text)
        primary_text = required_section if required_section else jd_text

        # Step 3: Regex scan on focused text
        regex_skills = self._regex_scan(primary_text)

        # Step 4: Full JD domain keyword scan (catches skills outside "Required" section)
        domain_skills = self._domain_keyword_scan(jd_text, already_found=set(regex_skills.keys()))

        # Merge regex + domain skills
        all_found = {**regex_skills, **domain_skills}

        # Step 5: Semantic scan (JD-optimized: fewer passes)
        semantic_skills = self._semantic_scan(
            primary_text,
            already_found=set(all_found.keys())
        )
        all_found.update(semantic_skills)

        # Step 6: Apply industry heuristics (boost/reweight)
        all_found = self._apply_heuristics(all_found, industry)

        # Sort by relevance score
        required_skills = sorted(
            all_found.values(),
            key=lambda x: x["relevance_score"],
            reverse=True
        )

        logger.info(f"[JDMiner] Extracted {len(required_skills)} required skills.")

        return {
            "required_skills": required_skills,
            "detected_industry": industry,
            "total_skills_found": len(required_skills),
            "jd_summary": {
                "from_required_section": bool(required_section),
                "regex_matches": len(regex_skills),
                "domain_matches": len(domain_skills),
                "semantic_matches": len(semantic_skills),
            }
        }

    # ─── Step 1: Industry Detection ───────────────────────────────────────────

    def _detect_industry(self, jd_lower: str) -> str:
        """
        Infer the industry from job description keywords.
        Returns the best-matching industry or 'tech' as default.
        """
        scores = {industry: 0 for industry in DOMAIN_INDICATORS}
        for industry, indicators in DOMAIN_INDICATORS.items():
            for indicator in indicators:
                if indicator in jd_lower:
                    scores[industry] += 1

        best_industry = max(scores, key=scores.get)
        return best_industry if scores[best_industry] > 0 else "tech"

    # ─── Step 2: Extract Required Skills Section ──────────────────────────────

    def _extract_required_section(self, jd_text: str) -> Optional[str]:
        """
        Try to isolate the "Required Skills" section from the JD.
        Uses multiple regex patterns as fallback chain.
        """
        for pattern in REQUIRED_SECTION_PATTERNS:
            match = re.search(pattern, jd_text)
            if match and len(match.group(1).strip()) > 20:
                section = match.group(1).strip()
                logger.debug(f"[JDMiner] Found required section ({len(section)} chars).")
                return section
        return None

    # ─── Step 3: Regex Scan ───────────────────────────────────────────────────

    def _regex_scan(self, text: str) -> Dict[str, Dict]:
        """Exact/near-exact skill name matching in text."""
        found: Dict[str, Dict] = {}
        text_lower = text.lower()

        for pattern, canonical in self.skill_patterns:
            if canonical in found:
                continue
            if pattern.search(text_lower):
                found[canonical] = self._build_result(canonical, relevance=1.0, method="regex")

        logger.debug(f"[JDMiner/Regex] Found {len(found)} skills.")
        return found

    # ─── Step 4: Domain Keyword Scan ─────────────────────────────────────────

    def _domain_keyword_scan(self, text: str, already_found: Set[str]) -> Dict[str, Dict]:
        """
        Fuzzy keyword scanning for domain-specific terms.
        Uses token_set_ratio with threshold 0.85.
        """
        found: Dict[str, Dict] = {}
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\+\#\-\.]*\b", text)

        # Only scan 1-gram and 2-gram in JD mode (faster)
        ngrams = []
        for n in range(1, 3):
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i:i + n]))

        for ngram in ngrams:
            ngram_lower = ngram.lower()
            for alias_lower, canonical in self.all_tokens:
                if canonical in already_found or canonical in found:
                    continue
                score = fuzz.token_set_ratio(ngram_lower, alias_lower) / 100.0
                if score >= FUZZY_THRESHOLD and score < 1.0:
                    found[canonical] = self._build_result(
                        canonical,
                        relevance=score * 0.85,
                        method="fuzzy"
                    )

        logger.debug(f"[JDMiner/Domain] Found {len(found)} additional skills.")
        return found

    # ─── Step 5: Semantic Scan (JD-Optimized) ─────────────────────────────────

    def _semantic_scan(self, text: str, already_found: Set[str]) -> Dict[str, Dict]:
        """
        SBERT semantic matching.
        JD-optimized: process only every 2nd sentence, max 20 sentences.
        """
        found: Dict[str, Dict] = {}

        # Split into sentences, sample in JD mode
        sentences = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 10]
        sentences = sentences[::2][:20]  # JD mode: every 2nd sentence, max 20

        if not sentences:
            return found

        # Encode sentences
        sent_embeddings = self.embedding_service.encode(sentences, normalize=True)

        # Get taxonomy embeddings
        skill_names, skill_embeddings = self.embedding_service.get_taxonomy_embeddings()

        for sent_emb in sent_embeddings:
            similarities = self.embedding_service.cosine_similarity(sent_emb, skill_embeddings)
            high_sim_indices = np.where(similarities >= SEMANTIC_THRESHOLD)[0]

            for idx in high_sim_indices:
                canonical = skill_names[idx]
                if canonical in already_found or canonical in found:
                    continue
                sim_score = float(similarities[idx])
                found[canonical] = self._build_result(
                    canonical,
                    relevance=sim_score * 0.80,
                    method="semantic"
                )

        logger.debug(f"[JDMiner/Semantic] Found {len(found)} additional skills.")
        return found

    # ─── Step 6: Industry Heuristics ─────────────────────────────────────────

    def _apply_heuristics(self, skills: Dict[str, Dict], industry: str) -> Dict[str, Dict]:
        """
        Apply industry-specific boosts to relevance scores.
        E.g., Java prioritized in finance contexts.
        """
        boosts = INDUSTRY_HEURISTICS.get(industry, {})
        for canonical, skill_data in skills.items():
            boost = boosts.get(canonical, 1.0)
            skill_data["relevance_score"] = min(
                round(skill_data["relevance_score"] * boost, 4),
                1.0
            )
            if boost > 1.0:
                skill_data["industry_boosted"] = True
                skill_data["industry"] = industry
        return skills

    # ─── Helper ───────────────────────────────────────────────────────────────

    def _build_result(self, canonical: str, relevance: float, method: str) -> Dict:
        """Construct skill result dict for JD mining output."""
        meta = self.category_map.get(canonical, {})
        return {
            "skill_name": canonical,
            "normalized_name": canonical.lower().replace(" ", "_").replace("/", "_"),
            "relevance_score": round(min(relevance, 1.0), 4),
            "extraction_method": method,
            "type": meta.get("type", "Technical"),
            "market_demand": meta.get("market_demand", 0.5),
            "difficulty": meta.get("difficulty", 3),
            "learning_hours": meta.get("learning_hours", 80),
            "industry_boosted": False,
        }
