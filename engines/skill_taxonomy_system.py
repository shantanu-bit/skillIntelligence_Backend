"""
Module C: Skill Taxonomy System
=================================
Manages the canonical skill taxonomy:
- Load and index skills from JSON
- Support dynamic updates (add/modify/delete)
- Precompute and cache SBERT embeddings
- Provide lookup by name, alias, category
- Expose taxonomy metadata (difficulty, market demand, salary impact)

Taxonomy data format per skill:
{
    "canonical_name": str,
    "aliases": List[str],
    "category": "Technical|Infrastructure|Soft|Domain-Specific",
    "difficulty": 1-5,
    "market_demand": 0-1,
    "salary_impact": 0-1,
    "learning_hours": int,
    "prerequisites": List[str],
    "embedding": null (populated at runtime)
}
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
import numpy as np

from utils.embedding_service import get_embedding_service
from config.settings import TAXONOMY_PATH, EMBEDDING_CACHE_PATH

logger = logging.getLogger(__name__)


class SkillTaxonomySystem:
    """
    Central skill taxonomy registry.
    Provides O(1) lookup, dynamic update, and embedding-enhanced queries.
    Inspired by ESCO, O*NET, and Kaggle job datasets.
    """

    def __init__(self, taxonomy_path: str = TAXONOMY_PATH):
        self.taxonomy_path = taxonomy_path
        self.embedding_service = get_embedding_service()

        # Primary data structures
        self._skills: List[Dict] = []
        self._by_canonical: Dict[str, Dict] = {}         # canonical_lower -> skill
        self._by_alias: Dict[str, str] = {}              # alias_lower -> canonical_name
        self._by_category: Dict[str, List[str]] = {}     # category -> [canonical_names]
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_names: List[str] = []

        self._load()
        logger.info(f"[Taxonomy] Loaded {len(self._skills)} skills from {taxonomy_path}")

    # ─── Load + Index ─────────────────────────────────────────────────────────

    def _load(self):
        """Load taxonomy and build all lookup structures."""
        with open(self.taxonomy_path, "r") as f:
            data = json.load(f)

        self._skills = data.get("skills", [])
        self._by_canonical = {}
        self._by_alias = {}
        self._by_category = {}

        for skill in self._skills:
            canonical = skill["canonical_name"]
            canonical_lower = canonical.lower()

            self._by_canonical[canonical_lower] = skill

            # Alias index
            for alias in [canonical] + skill.get("aliases", []):
                self._by_alias[alias.lower()] = canonical

            # Category index
            category = skill.get("category", "Technical")
            if category not in self._by_category:
                self._by_category[category] = []
            self._by_category[category].append(canonical)

        # Load or build embeddings
        names, embeddings = self.embedding_service.get_taxonomy_embeddings()
        self._embedding_names = names
        self._embeddings = embeddings

    # ─── Lookup Methods ───────────────────────────────────────────────────────

    def get_skill(self, name: str) -> Optional[Dict]:
        """
        Look up a skill by canonical name or any alias (case-insensitive).

        Args:
            name: Skill name or alias

        Returns:
            Full skill dict or None if not found
        """
        name_lower = name.lower()

        # Direct canonical lookup
        if name_lower in self._by_canonical:
            return self._by_canonical[name_lower]

        # Alias lookup -> resolve to canonical -> full skill
        if name_lower in self._by_alias:
            canonical = self._by_alias[name_lower]
            return self._by_canonical.get(canonical.lower())

        return None

    def get_canonical_name(self, name: str) -> Optional[str]:
        """Resolve any skill name or alias to its canonical form."""
        name_lower = name.lower()
        if name_lower in self._by_alias:
            return self._by_alias[name_lower]
        if name_lower in self._by_canonical:
            return self._by_canonical[name_lower]["canonical_name"]
        return None

    def get_by_category(self, category: str) -> List[Dict]:
        """
        Retrieve all skills in a given category.

        Args:
            category: "Technical", "Infrastructure", "Soft", "Domain-Specific"
        """
        canonical_names = self._by_category.get(category, [])
        return [self._by_canonical[n.lower()] for n in canonical_names if n.lower() in self._by_canonical]

    def get_all_skills(self) -> List[Dict]:
        """Return all skills in the taxonomy."""
        return self._skills.copy()

    def get_all_canonical_names(self) -> List[str]:
        """Return list of all canonical skill names."""
        return [s["canonical_name"] for s in self._skills]

    def get_prerequisites(self, skill_name: str) -> List[str]:
        """Return prerequisites for a skill."""
        skill = self.get_skill(skill_name)
        if skill:
            return skill.get("prerequisites", [])
        return []

    def resolve_prerequisites_tree(self, skill_name: str, depth: int = 0, max_depth: int = 5) -> Dict:
        """
        Recursively build the full prerequisite dependency tree for a skill.

        Args:
            skill_name: Root skill to analyze
            depth: Current recursion depth
            max_depth: Maximum recursion depth to prevent cycles

        Returns:
            Nested dict representing dependency tree
        """
        if depth >= max_depth:
            return {"name": skill_name, "prerequisites": [], "truncated": True}

        skill = self.get_skill(skill_name)
        if not skill:
            return {"name": skill_name, "prerequisites": [], "not_in_taxonomy": True}

        prerequisites = skill.get("prerequisites", [])
        tree = {
            "name": skill_name,
            "difficulty": skill.get("difficulty", 3),
            "learning_hours": skill.get("learning_hours", 0),
            "prerequisites": [
                self.resolve_prerequisites_tree(prereq, depth + 1, max_depth)
                for prereq in prerequisites
            ]
        }
        return tree

    # ─── Dynamic Updates ──────────────────────────────────────────────────────

    def add_skill(self, skill: Dict) -> bool:
        """
        Dynamically add a new skill to the taxonomy.
        Persists to JSON and invalidates embedding cache.

        Args:
            skill: Dict conforming to taxonomy schema

        Returns:
            True if added, False if skill already exists
        """
        canonical = skill.get("canonical_name", "")
        if not canonical:
            logger.error("[Taxonomy] Cannot add skill without canonical_name.")
            return False

        if self.get_skill(canonical):
            logger.warning(f"[Taxonomy] Skill '{canonical}' already exists. Use update_skill() instead.")
            return False

        # Assign an ID
        existing_ids = [int(s.get("id", "SK000")[2:]) for s in self._skills if s.get("id", "").startswith("SK")]
        new_id = f"SK{max(existing_ids, default=0) + 1:03d}"
        skill["id"] = new_id

        # Ensure required fields
        skill.setdefault("aliases", [])
        skill.setdefault("category", "Technical")
        skill.setdefault("difficulty", 3)
        skill.setdefault("market_demand", 0.5)
        skill.setdefault("salary_impact", 0.5)
        skill.setdefault("learning_hours", 80)
        skill.setdefault("prerequisites", [])
        skill.setdefault("embedding", None)

        self._skills.append(skill)
        self._rebuild_index()
        self._persist()
        self.embedding_service.invalidate_cache()

        logger.info(f"[Taxonomy] Added new skill: {canonical} (ID: {new_id})")
        return True

    def update_skill(self, canonical_name: str, updates: Dict) -> bool:
        """
        Update fields of an existing skill.

        Args:
            canonical_name: The canonical name of the skill to update
            updates: Dict of fields to update

        Returns:
            True if updated, False if not found
        """
        skill = self.get_skill(canonical_name)
        if not skill:
            logger.warning(f"[Taxonomy] Skill '{canonical_name}' not found for update.")
            return False

        skill.update(updates)
        self._rebuild_index()
        self._persist()

        # Invalidate embeddings if aliases or name changed
        if "canonical_name" in updates or "aliases" in updates:
            self.embedding_service.invalidate_cache()

        logger.info(f"[Taxonomy] Updated skill: {canonical_name}")
        return True

    def delete_skill(self, canonical_name: str) -> bool:
        """Remove a skill from the taxonomy."""
        skill = self.get_skill(canonical_name)
        if not skill:
            logger.warning(f"[Taxonomy] Skill '{canonical_name}' not found for deletion.")
            return False

        self._skills = [s for s in self._skills if s["canonical_name"] != skill["canonical_name"]]
        self._rebuild_index()
        self._persist()
        self.embedding_service.invalidate_cache()

        logger.info(f"[Taxonomy] Deleted skill: {canonical_name}")
        return True

    def _rebuild_index(self):
        """Rebuild all lookup dictionaries after a modification."""
        self._by_canonical = {}
        self._by_alias = {}
        self._by_category = {}

        for skill in self._skills:
            canonical = skill["canonical_name"]
            self._by_canonical[canonical.lower()] = skill

            for alias in [canonical] + skill.get("aliases", []):
                self._by_alias[alias.lower()] = canonical

            category = skill.get("category", "Technical")
            if category not in self._by_category:
                self._by_category[category] = []
            self._by_category[category].append(canonical)

    def _persist(self):
        """Save current taxonomy state to JSON file."""
        data = {
            "version": "1.0.0",
            "source": "ESCO + O*NET + Kaggle Job Datasets",
            "skills": self._skills
        }
        with open(self.taxonomy_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"[Taxonomy] Persisted {len(self._skills)} skills to {self.taxonomy_path}")

    # ─── Market Metrics ───────────────────────────────────────────────────────

    def get_market_stats(self) -> Dict:
        """Return aggregated market statistics across the taxonomy."""
        if not self._skills:
            return {}

        demands = [s.get("market_demand", 0) for s in self._skills]
        salaries = [s.get("salary_impact", 0) for s in self._skills]
        difficulties = [s.get("difficulty", 3) for s in self._skills]

        return {
            "total_skills": len(self._skills),
            "categories": list(self._by_category.keys()),
            "avg_market_demand": round(sum(demands) / len(demands), 3),
            "avg_salary_impact": round(sum(salaries) / len(salaries), 3),
            "avg_difficulty": round(sum(difficulties) / len(difficulties), 2),
            "high_demand_skills": [
                s["canonical_name"] for s in self._skills
                if s.get("market_demand", 0) >= 0.85
            ],
        }

    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Find skills semantically similar to a query using SBERT.
        Useful for discovering related skills.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of (skill_name, similarity_score) dicts
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            names, embeddings = self.embedding_service.get_taxonomy_embeddings()
            self._embedding_names = names
            self._embeddings = embeddings

        query_embedding = self.embedding_service.encode([query], normalize=True)[0]
        similarities = self.embedding_service.cosine_similarity(query_embedding, self._embeddings)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            skill_name = self._embedding_names[idx]
            skill = self.get_skill(skill_name)
            if skill:
                results.append({
                    "skill_name": skill_name,
                    "similarity": round(float(similarities[idx]), 4),
                    "category": skill.get("category"),
                    "difficulty": skill.get("difficulty"),
                })
        return results

    # ─── Schema Summary ───────────────────────────────────────────────────────

    def describe(self) -> str:
        """Return a human-readable summary of the taxonomy."""
        stats = self.get_market_stats()
        lines = [
            "=== Skill Taxonomy Summary ===",
            f"Total Skills: {stats.get('total_skills', 0)}",
            f"Categories: {', '.join(stats.get('categories', []))}",
            f"Avg Market Demand: {stats.get('avg_market_demand', 0):.1%}",
            f"Avg Salary Impact: {stats.get('avg_salary_impact', 0):.1%}",
            f"Avg Difficulty: {stats.get('avg_difficulty', 0):.1f}/5",
            f"High-Demand Skills: {', '.join(stats.get('high_demand_skills', [])[:5])}...",
        ]
        return "\n".join(lines)
