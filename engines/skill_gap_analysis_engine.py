"""
Module E: Skill Gap Analysis Engine
=====================================
Compares extracted resume skills vs required JD skills to compute gaps.

For each missing skill:
    Gap Score = (Market Demand × Salary Impact) / (Difficulty × Learning Hours)

Returns:
    - Total gaps
    - Weighted average severity
    - Match percentage
    - Average difficulty of gaps
    - Dependency-aware ordered skill acquisition sequence
"""

import logging
import math
from typing import List, Dict, Set, Tuple

logger = logging.getLogger(__name__)


class SkillGapAnalysisEngine:
    """
    Analyzes the gap between candidate skills (from resume) and required
    skills (from job description), with dependency-aware acquisition ordering.
    """

    def __init__(self):
        logger.info("[GapAnalysis] Engine initialized.")

    # ─── Main Interface ───────────────────────────────────────────────────────

    def analyze(
        self,
        resume_skills: List[Dict],
        jd_skills: List[Dict]
    ) -> Dict:
        """
        Perform full gap analysis between resume and JD skills.

        Args:
            resume_skills: Skills extracted from resume (from HybridSkillExtractionEngine)
            jd_skills: Required skills from job description (from JDSkillMiner)

        Returns:
            Comprehensive gap analysis report dict
        """
        if not jd_skills:
            return {"error": "No JD skills provided for comparison."}

        # Build sets for comparison
        resume_canonical: Set[str] = {
            s["skill_name"].lower() for s in resume_skills
        }
        jd_canonical: Set[str] = {
            s["skill_name"].lower() for s in jd_skills
        }

        # Matched skills (present in both)
        matched = jd_canonical.intersection(resume_canonical)

        # Missing skills (required but not in resume)
        missing_names = jd_canonical - resume_canonical

        # Build lookup for JD skill metadata
        jd_lookup: Dict[str, Dict] = {
            s["skill_name"].lower(): s for s in jd_skills
        }

        # Build lookup for resume skill strength scores
        resume_strength: Dict[str, float] = {
            s["skill_name"].lower(): s.get("strength_score", 0.5)
            for s in resume_skills
        }

        # ─── Compute Gap Metrics ──────────────────────────────────────────────

        gap_skills = []
        for skill_name in missing_names:
            skill_data = jd_lookup.get(skill_name, {})
            gap_entry = self._compute_gap_entry(skill_name, skill_data)
            gap_skills.append(gap_entry)

        # Sort gaps by gap_score descending (highest ROI first)
        gap_skills.sort(key=lambda x: x["gap_score"], reverse=True)

        # ─── Aggregate Statistics ─────────────────────────────────────────────

        total_required = len(jd_canonical)
        total_matched = len(matched)
        total_gaps = len(gap_skills)
        match_percentage = round((total_matched / total_required) * 100, 1) if total_required else 0

        # Weighted average severity (weighted by market_demand)
        if gap_skills:
            total_weight = sum(g["market_demand"] for g in gap_skills)
            if total_weight > 0:
                weighted_severity = sum(
                    g["gap_score"] * g["market_demand"] for g in gap_skills
                ) / total_weight
            else:
                weighted_severity = sum(g["gap_score"] for g in gap_skills) / len(gap_skills)
            weighted_severity = round(weighted_severity, 4)

            avg_difficulty = round(
                sum(g["difficulty"] for g in gap_skills) / len(gap_skills), 2
            )
        else:
            weighted_severity = 0.0
            avg_difficulty = 0.0

        # ─── Dependency-Aware Acquisition Sequence ────────────────────────────

        acquisition_sequence = self._compute_acquisition_sequence(
            gap_skills, resume_canonical
        )

        # ─── Profile matched skills with strength ─────────────────────────────

        matched_skill_details = []
        for skill_name in matched:
            jd_info = jd_lookup.get(skill_name, {})
            matched_skill_details.append({
                "skill_name": jd_info.get("skill_name", skill_name.title()),
                "resume_strength": round(resume_strength.get(skill_name, 0.5), 3),
                "relevance_score": jd_info.get("relevance_score", 0.5),
                "type": jd_info.get("type", "Technical"),
            })

        return {
            "summary": {
                "total_required": total_required,
                "total_matched": total_matched,
                "total_gaps": total_gaps,
                "match_percentage": match_percentage,
                "weighted_avg_severity": weighted_severity,
                "average_gap_difficulty": avg_difficulty,
                "profile_strength": self._compute_profile_strength(
                    match_percentage, weighted_severity
                ),
            },
            "matched_skills": sorted(
                matched_skill_details,
                key=lambda x: x["relevance_score"],
                reverse=True
            ),
            "skill_gaps": gap_skills,
            "acquisition_sequence": acquisition_sequence,
        }

    # ─── Gap Score Computation ────────────────────────────────────────────────

    def _compute_gap_entry(self, skill_name: str, skill_data: Dict) -> Dict:
        """
        Compute gap analysis metrics for a single missing skill.

        Gap Score = (Market Demand × Salary Impact) / (Difficulty × Learning Hours)

        Higher gap_score = Higher ROI for learning this skill

        Args:
            skill_name: Lowercase canonical skill name
            skill_data: Skill metadata from JD mining

        Returns:
            Gap entry dict
        """
        market_demand = skill_data.get("market_demand", 0.5)
        salary_impact = skill_data.get("salary_impact", skill_data.get("market_demand", 0.5))
        difficulty = max(skill_data.get("difficulty", 3), 1)  # Avoid division by zero
        learning_hours = max(skill_data.get("learning_hours", 80), 1)

        # Normalize learning hours to 0-1 scale (cap at 300 hours)
        normalized_hours = min(learning_hours / 300.0, 1.0)
        normalized_difficulty = difficulty / 5.0

        # Gap Score formula
        numerator = market_demand * salary_impact
        denominator = normalized_difficulty * normalized_hours

        if denominator == 0:
            gap_score = 0.0
        else:
            gap_score = round(numerator / denominator, 4)

        # Severity: how critical is this gap (0=low, 1=critical)
        severity = round(market_demand * 0.6 + salary_impact * 0.4, 4)

        return {
            "skill_name": skill_data.get("skill_name", skill_name.title()),
            "normalized_name": skill_name.replace(" ", "_"),
            "gap_score": gap_score,
            "severity": severity,
            "market_demand": market_demand,
            "salary_impact": salary_impact,
            "difficulty": difficulty,
            "learning_hours": learning_hours,
            "type": skill_data.get("type", "Technical"),
            "prerequisites": skill_data.get("prerequisites", []),
            "extraction_method": skill_data.get("extraction_method", "regex"),
        }

    # ─── Profile Strength ─────────────────────────────────────────────────────

    def _compute_profile_strength(self, match_pct: float, severity: float) -> str:
        """
        Compute overall profile strength label based on match % and gap severity.

        Returns:
            One of: "Excellent", "Good", "Moderate", "Needs Work", "Significant Gap"
        """
        if match_pct >= 85:
            return "Excellent"
        elif match_pct >= 70:
            return "Good"
        elif match_pct >= 50:
            return "Moderate"
        elif match_pct >= 30:
            return "Needs Work"
        else:
            return "Significant Gap"

    # ─── Dependency-Aware Acquisition Sequence ────────────────────────────────

    def _compute_acquisition_sequence(
        self,
        gap_skills: List[Dict],
        already_known: Set[str]
    ) -> List[Dict]:
        """
        Order gap skills for acquisition considering:
        1. Prerequisites must come before dependent skills
        2. Within same dependency level, higher gap_score first (ROI ordering)

        Uses topological sort (Kahn's algorithm variant).

        Args:
            gap_skills: List of gap skill dicts
            already_known: Set of lowercase skill names already in the resume

        Returns:
            Ordered list of skills to learn, with sequence metadata
        """
        if not gap_skills:
            return []

        # Build a lookup for gap skills
        gap_names = {s["normalized_name"]: s for s in gap_skills}
        gap_canonical = {s["skill_name"].lower(): s for s in gap_skills}

        # Build adjacency list: skill -> [skills it enables]
        # and in-degree counter
        in_degree: Dict[str, int] = {s["skill_name"]: 0 for s in gap_skills}
        dependents: Dict[str, List[str]] = {s["skill_name"]: [] for s in gap_skills}

        for skill in gap_skills:
            prereqs = skill.get("prerequisites", [])
            for prereq in prereqs:
                prereq_lower = prereq.lower()
                # Only count prerequisites that are also in the gap list
                # (not counting prereqs already known)
                if prereq_lower in gap_canonical and prereq_lower not in already_known:
                    prereq_canonical = gap_canonical[prereq_lower]["skill_name"]
                    in_degree[skill["skill_name"]] += 1
                    dependents[prereq_canonical].append(skill["skill_name"])

        # Kahn's BFS topological sort
        # Start with skills that have no unmet prerequisites
        from collections import deque

        queue = deque()
        for skill in gap_skills:
            if in_degree[skill["skill_name"]] == 0:
                queue.append(skill["skill_name"])

        # Within each "ready" tier, sort by gap_score (highest ROI first)
        sequence: List[Dict] = []
        step = 1

        while queue:
            # Process all skills in current tier
            current_tier = sorted(
                [n for n in queue],
                key=lambda n: gap_canonical[n.lower()].get("gap_score", 0),
                reverse=True
            )
            queue.clear()

            for skill_name in current_tier:
                skill_data = gap_canonical.get(skill_name.lower(), {})
                sequence.append({
                    "step": step,
                    "skill_name": skill_name,
                    "gap_score": skill_data.get("gap_score", 0),
                    "difficulty": skill_data.get("difficulty", 3),
                    "learning_hours": skill_data.get("learning_hours", 80),
                    "type": skill_data.get("type", "Technical"),
                    "prerequisites_met": True,
                    "reason": self._explain_priority(skill_data, step),
                })
                step += 1

                # Unlock dependent skills
                for dependent in dependents.get(skill_name, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Handle any remaining skills (cycles in prerequisites, if any)
        remaining = [
            s for s in gap_skills
            if not any(seq["skill_name"] == s["skill_name"] for seq in sequence)
        ]
        for skill in remaining:
            sequence.append({
                "step": step,
                "skill_name": skill["skill_name"],
                "gap_score": skill.get("gap_score", 0),
                "difficulty": skill.get("difficulty", 3),
                "learning_hours": skill.get("learning_hours", 80),
                "type": skill.get("type", "Technical"),
                "prerequisites_met": False,
                "reason": "Circular dependency detected; review prerequisites",
            })
            step += 1

        return sequence

    def _explain_priority(self, skill: Dict, step: int) -> str:
        """Generate human-readable explanation for why a skill is prioritized."""
        gap_score = skill.get("gap_score", 0)
        difficulty = skill.get("difficulty", 3)
        market_demand = skill.get("market_demand", 0.5)

        if step <= 3 and gap_score >= 1.0:
            return "High ROI: high market demand + strong salary impact with manageable difficulty"
        elif market_demand >= 0.85:
            return f"Critical market demand ({market_demand:.0%}): essential for role fit"
        elif difficulty <= 2:
            return "Quick win: low difficulty enables fast proficiency"
        elif skill.get("prerequisites", []):
            return "Prerequisite skill: unlocks higher-level capabilities"
        else:
            return f"Gap score {gap_score:.2f}: prioritized by ROI formula"
