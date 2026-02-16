"""
Module B: Hybrid Skill Extraction Engine
==========================================
3-Layer extraction pipeline:

Layer 1 - Regex-Based Extraction:
    Match known skills from taxonomy using word boundaries.
    High precision, low recall.

Layer 2 - Fuzzy Matching + Alias Detection:
    token_set_ratio threshold = 0.85
    Handles abbreviations and aliases.

Layer 3 - Semantic Skill Identification:
    Sentence-BERT (all-MiniLM-L6-v2) cosine similarity threshold = 0.75
    Detects implicit, contextual and tacit skills.

Output per skill:
{
    "skill_name": str,
    "normalized_name": str,
    "strength_score": float (0-1),
    "extraction_method": "regex|fuzzy|semantic",
    "type": "Technical|Soft|Domain"
}
"""

import re
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from utils.compat import fuzz

from utils.embedding_service import get_embedding_service
from config.settings import (
    FUZZY_THRESHOLD,
    SEMANTIC_THRESHOLD,
    LAYER_WEIGHTS,
    TAXONOMY_PATH,
)

logger = logging.getLogger(__name__)


class HybridSkillExtractionEngine:
    """
    Three-layer hybrid skill extractor that combines:
    - Deterministic regex matching (precision)
    - Fuzzy alias matching (recall boost)
    - Semantic SBERT matching (implicit/tacit skill detection)
    """

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.taxonomy = self._load_taxonomy()
        self._build_lookup_structures()
        logger.info("[SkillExtraction] Engine initialized.")

    # ─── Initialization ───────────────────────────────────────────────────────

    def _load_taxonomy(self) -> List[Dict]:
        """Load skill taxonomy from JSON."""
        with open(TAXONOMY_PATH, "r") as f:
            data = json.load(f)
        return data["skills"]

    def _build_lookup_structures(self):
        """
        Pre-build efficient data structures for Layer 1 and Layer 2.
        - Regex patterns compiled from canonical names + aliases
        - Alias map: alias_lower -> canonical_name
        - Category map: canonical_name -> category info
        """
        self.alias_map: Dict[str, str] = {}       # alias_lower -> canonical_name
        self.category_map: Dict[str, Dict] = {}   # canonical_name -> metadata
        self.all_skill_tokens: List[Tuple[str, str]] = []  # (token_lower, canonical_name)

        for skill in self.taxonomy:
            canonical = skill["canonical_name"]
            category_info = {
                "type": skill.get("category", "Technical"),
                "difficulty": skill.get("difficulty", 3),
                "market_demand": skill.get("market_demand", 0.5),
                "salary_impact": skill.get("salary_impact", 0.5),
                "learning_hours": skill.get("learning_hours", 80),
                "prerequisites": skill.get("prerequisites", []),
            }
            self.category_map[canonical] = category_info

            # Add canonical name and all aliases to lookup maps
            all_names = [canonical] + skill.get("aliases", [])
            for name in all_names:
                name_lower = name.lower()
                self.alias_map[name_lower] = canonical
                self.all_skill_tokens.append((name_lower, canonical))

        # Build compiled regex patterns (Layer 1)
        # Sort by length (longer patterns first to avoid partial matches)
        self.regex_patterns: List[Tuple[re.Pattern, str]] = []
        for alias_lower, canonical in sorted(
            self.all_skill_tokens, key=lambda x: len(x[0]), reverse=True
        ):
            # Escape special regex chars in skill names
            escaped = re.escape(alias_lower)
            pattern = re.compile(
                rf"\b{escaped}\b",
                flags=re.IGNORECASE
            )
            self.regex_patterns.append((pattern, canonical))

        logger.info(
            f"[SkillExtraction] Built lookup: "
            f"{len(self.alias_map)} aliases, {len(self.regex_patterns)} regex patterns."
        )

    # ─── Main Interface ───────────────────────────────────────────────────────

    def extract_skills(self, text: str, mode: str = "resume") -> List[Dict]:
        """
        Run all three extraction layers and merge results.

        Args:
            text: Normalized text to analyze
            mode: "resume" (exhaustive) or "jd" (faster, fewer semantic passes)

        Returns:
            List of skill dicts sorted by strength_score descending
        """
        if not text.strip():
            return []

        logger.info(f"[SkillExtraction] Extracting skills (mode={mode}), text length={len(text)}")

        # Accumulated skills: canonical_name -> best extraction result
        skill_registry: Dict[str, Dict] = {}

        # Layer 1: Regex
        layer1_results = self._layer1_regex(text)
        for result in layer1_results:
            self._register_skill(skill_registry, result)

        # Layer 2: Fuzzy
        layer2_results = self._layer2_fuzzy(text, already_found={r["skill_name"] for r in layer1_results})
        for result in layer2_results:
            self._register_skill(skill_registry, result)

        # Layer 3: Semantic (fewer passes in JD mode)
        layer3_results = self._layer3_semantic(
            text,
            already_found=set(skill_registry.keys()),
            mode=mode
        )
        for result in layer3_results:
            self._register_skill(skill_registry, result)

        # Sort by strength_score
        final_skills = sorted(
            skill_registry.values(),
            key=lambda x: x["strength_score"],
            reverse=True
        )

        logger.info(
            f"[SkillExtraction] Found {len(final_skills)} skills: "
            f"L1={len(layer1_results)}, L2={len(layer2_results)}, L3={len(layer3_results)}"
        )
        return final_skills

    def _register_skill(self, registry: Dict, skill: Dict):
        """
        Register or update skill in the registry.
        If skill already exists, keep the highest strength_score.
        """
        name = skill["skill_name"]
        if name not in registry:
            registry[name] = skill
        else:
            # Keep max strength, record all methods used
            existing = registry[name]
            if skill["strength_score"] > existing["strength_score"]:
                registry[name] = skill
            else:
                # Merge extraction method info
                existing_method = existing["extraction_method"]
                new_method = skill["extraction_method"]
                if new_method not in existing_method:
                    existing["extraction_method"] = f"{existing_method}+{new_method}"

    # ─── Layer 1: Regex-Based Extraction ─────────────────────────────────────

    def _layer1_regex(self, text: str) -> List[Dict]:
        """
        Exact/near-exact matching using compiled regex patterns.
        High precision - only matches known skill names and aliases.
        """
        found: Set[str] = set()
        results: List[Dict] = []

        text_lower = text.lower()

        for pattern, canonical in self.regex_patterns:
            if canonical in found:
                continue  # Already found this skill

            if pattern.search(text_lower):
                found.add(canonical)
                results.append(self._build_skill_result(
                    canonical=canonical,
                    strength=LAYER_WEIGHTS["regex"],
                    method="regex"
                ))

        logger.debug(f"[Layer1/Regex] Found {len(results)} skills.")
        return results

    # ─── Layer 2: Fuzzy Matching + Alias Detection ────────────────────────────

    def _layer2_fuzzy(self, text: str, already_found: Set[str]) -> List[Dict]:
        """
        Fuzzy token-set ratio matching for aliases, abbreviations, and typos.
        Threshold: 0.85 (85 out of 100 on RapidFuzz scale).
        """
        results: List[Dict] = []

        # Tokenize text into candidate n-grams for matching
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\+\#\-\.]*\b", text)

        # Generate 1-gram, 2-gram, 3-gram candidates
        ngrams = self._generate_ngrams(words, max_n=3)

        found_in_layer: Set[str] = set()

        for ngram in ngrams:
            ngram_lower = ngram.lower()

            for alias_lower, canonical in self.all_skill_tokens:
                if canonical in already_found or canonical in found_in_layer:
                    continue

                # token_set_ratio handles word order differences (e.g., "Learning Machine")
                score = fuzz.token_set_ratio(ngram_lower, alias_lower) / 100.0

                if score >= FUZZY_THRESHOLD and score < 1.0:  # < 1.0 = not exact (would be in Layer 1)
                    found_in_layer.add(canonical)
                    # Scale strength by fuzzy score
                    strength = LAYER_WEIGHTS["fuzzy"] * score
                    results.append(self._build_skill_result(
                        canonical=canonical,
                        strength=strength,
                        method="fuzzy"
                    ))
                    break  # Found this ngram's best match

        logger.debug(f"[Layer2/Fuzzy] Found {len(results)} new skills.")
        return results

    def _generate_ngrams(self, words: List[str], max_n: int = 3) -> List[str]:
        """Generate n-grams from word list."""
        ngrams = []
        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                ngrams.append(ngram)
        return ngrams

    # ─── Layer 3: Semantic Skill Identification ───────────────────────────────

    def _layer3_semantic(
        self,
        text: str,
        already_found: Set[str],
        mode: str = "resume"
    ) -> List[Dict]:
        """
        Semantic matching using Sentence-BERT embeddings.
        Detects implicit, contextual, and tacit skills.
        Threshold: cosine similarity >= 0.75

        JD mode: processes fewer sentences (top-level scan only)
        Resume mode: exhaustive sentence-by-sentence analysis

        Example:
            "optimized distributed data pipelines" →
            Detects: Apache Spark, Distributed Systems
        """
        results: List[Dict] = []

        # Split text into sentences for semantic analysis
        sentences = self._split_sentences(text)

        if mode == "jd":
            # JD mode: sample every other sentence, max 30 sentences
            sentences = sentences[::2][:30]
        else:
            # Resume mode: all sentences, max 60
            sentences = sentences[:60]

        if not sentences:
            return []

        # Compute sentence embeddings
        sentence_embeddings = self.embedding_service.encode(sentences, normalize=True)

        # Get precomputed taxonomy embeddings
        skill_names, skill_embeddings = self.embedding_service.get_taxonomy_embeddings()

        found_in_layer: Set[str] = set()

        for sentence, sent_embedding in zip(sentences, sentence_embeddings):
            # Compute cosine similarity with all skill embeddings
            similarities = self.embedding_service.cosine_similarity(sent_embedding, skill_embeddings)

            # Find skills above threshold
            high_sim_indices = np.where(similarities >= SEMANTIC_THRESHOLD)[0]

            for idx in high_sim_indices:
                canonical = skill_names[idx]
                if canonical in already_found or canonical in found_in_layer:
                    continue

                sim_score = float(similarities[idx])
                found_in_layer.add(canonical)
                strength = LAYER_WEIGHTS["semantic"] * sim_score

                results.append(self._build_skill_result(
                    canonical=canonical,
                    strength=strength,
                    method="semantic",
                    context=sentence[:100]  # Attach triggering sentence for explainability
                ))

        logger.debug(f"[Layer3/Semantic] Found {len(results)} new skills.")
        return results

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into meaningful sentence chunks for semantic analysis.
        Handles resume bullet points and multi-line content.
        """
        # Split on sentence endings, newlines, and bullet-style breaks
        raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences = []
        for s in raw_sentences:
            s = s.strip()
            if len(s) > 10:  # Filter trivially short fragments
                sentences.append(s)
        return sentences

    # ─── Helper: Build Skill Result Dict ─────────────────────────────────────

    def _build_skill_result(
        self,
        canonical: str,
        strength: float,
        method: str,
        context: Optional[str] = None
    ) -> Dict:
        """
        Construct a standardized skill output dict.

        Args:
            canonical: The canonical skill name from taxonomy
            strength: Aggregated confidence score (0-1)
            method: Which layer detected it (regex/fuzzy/semantic)
            context: Optional triggering text context

        Returns:
            Skill result dict matching the specified output format
        """
        meta = self.category_map.get(canonical, {})
        skill_type = meta.get("type", "Technical")

        # Map category to output type labels
        type_label_map = {
            "Technical": "Technical",
            "Infrastructure": "Technical",
            "Soft": "Soft",
            "Domain-Specific": "Domain",
        }
        type_label = type_label_map.get(skill_type, "Technical")

        result = {
            "skill_name": canonical,
            "normalized_name": canonical.lower().replace(" ", "_").replace("/", "_"),
            "strength_score": round(min(strength, 1.0), 4),
            "extraction_method": method,
            "type": type_label,
            "difficulty": meta.get("difficulty", 3),
            "market_demand": meta.get("market_demand", 0.5),
            "salary_impact": meta.get("salary_impact", 0.5),
            "learning_hours": meta.get("learning_hours", 80),
            "prerequisites": meta.get("prerequisites", []),
        }

        if context:
            result["detected_context"] = context

        return result
