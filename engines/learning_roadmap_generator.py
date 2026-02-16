"""
Module F: Learning Roadmap Generator
======================================
Generates a structured, tiered learning roadmap from skill gap analysis.

3-Tier Prioritization:
    Tier 1: High ROI + Low Difficulty  (Quick wins + critical fundamentals)
    Tier 2: Medium ROI + Moderate Difficulty  (Core competency building)
    Tier 3: Low ROI but Necessary  (Specialized/advanced topics)

For each skill:
    - Weeks to proficiency
    - Learning hours breakdown
    - Prerequisites
    - Recommended learning type
    - Quick-win motivational ordering

Respects dependency ordering (e.g., Python → Django)
"""

import logging
from typing import List, Dict, Optional
import math

logger = logging.getLogger(__name__)

# ─── Learning Type Heuristics ─────────────────────────────────────────────────
# Recommend learning format based on skill type and difficulty
LEARNING_TYPE_MAP = {
    ("Technical", 1): "Interactive tutorial + hands-on projects",
    ("Technical", 2): "Online course + coding exercises",
    ("Technical", 3): "Structured course + real-world projects + documentation",
    ("Technical", 4): "Deep-dive specialization + research papers + mentorship",
    ("Technical", 5): "Advanced course + research + open-source contribution",
    ("Infrastructure", 1): "Documentation + sandbox environment",
    ("Infrastructure", 2): "Lab-based course + cloud free tier practice",
    ("Infrastructure", 3): "Certification path + lab exercises",
    ("Infrastructure", 4): "Certification + enterprise projects",
    ("Infrastructure", 5): "Expert-led training + production experience",
    ("Soft", 1): "Self-study + daily practice",
    ("Soft", 2): "Workshop + peer feedback",
    ("Soft", 3): "Coaching + structured practice",
    ("Soft", 4): "Long-term mentorship + leadership roles",
    ("Soft", 5): "Executive coaching + sustained leadership practice",
    ("Domain", 1): "Industry articles + introductory courses",
    ("Domain", 2): "Domain-specific online course + case studies",
    ("Domain", 3): "Specialization course + domain projects",
    ("Domain", 4): "Advanced specialization + industry certifications",
    ("Domain", 5): "Expert-led training + industry immersion",
}


class LearningRoadmapGenerator:
    """
    Converts skill gap analysis results into a structured,
    actionable learning roadmap with motivational quick-win ordering.
    """

    def __init__(self):
        logger.info("[Roadmap] Generator initialized.")

    # ─── Main Interface ───────────────────────────────────────────────────────

    def generate(
        self,
        gap_analysis: Dict,
        weekly_hours: int = 10,
        target_weeks: Optional[int] = None
    ) -> Dict:
        """
        Generate a complete learning roadmap from gap analysis results.

        Args:
            gap_analysis: Output from SkillGapAnalysisEngine.analyze()
            weekly_hours: How many hours per week the learner can dedicate
            target_weeks: Optional target completion timeline

        Returns:
            Full roadmap dict with tiers, timeline, milestones
        """
        skill_gaps = gap_analysis.get("skill_gaps", [])
        acquisition_sequence = gap_analysis.get("acquisition_sequence", [])

        if not skill_gaps:
            return {
                "message": "No skill gaps detected! Your profile is a strong match.",
                "tiers": [],
                "timeline": {"total_weeks": 0, "total_hours": 0},
                "milestones": []
            }

        logger.info(f"[Roadmap] Generating roadmap for {len(skill_gaps)} skill gaps.")

        # Build lookup from acquisition sequence for ordering
        sequence_order = {
            item["skill_name"]: item["step"]
            for item in acquisition_sequence
        }

        # Categorize skills into tiers
        tiers = self._categorize_into_tiers(skill_gaps, sequence_order)

        # Build individual skill roadmap entries
        all_entries = []
        for tier_name, tier_skills in tiers.items():
            for skill in tier_skills:
                entry = self._build_roadmap_entry(
                    skill=skill,
                    tier=tier_name,
                    weekly_hours=weekly_hours
                )
                all_entries.append(entry)

        # Compute cumulative timeline
        all_entries = self._assign_timeline(all_entries, weekly_hours)

        # Separate back into tiers for structured output
        tier_output = {"tier_1": [], "tier_2": [], "tier_3": []}
        for entry in all_entries:
            tier = entry.get("tier", "tier_2")
            if tier in tier_output:
                tier_output[tier].append(entry)

        # Compute overall stats
        total_hours = sum(e["learning_hours"] for e in all_entries)
        total_weeks = math.ceil(total_hours / max(weekly_hours, 1))

        # Generate milestones
        milestones = self._generate_milestones(tier_output, total_weeks)

        # Build quick-win path (first 3 items across all tiers)
        quick_wins = self._extract_quick_wins(tier_output)

        return {
            "roadmap_summary": {
                "total_skills_to_learn": len(skill_gaps),
                "total_hours_required": total_hours,
                "total_weeks_estimated": total_weeks,
                "weekly_commitment_hours": weekly_hours,
                "tier_1_count": len(tier_output["tier_1"]),
                "tier_2_count": len(tier_output["tier_2"]),
                "tier_3_count": len(tier_output["tier_3"]),
            },
            "quick_wins": quick_wins,
            "tiers": {
                "tier_1": {
                    "name": "High ROI + Low Difficulty",
                    "description": "Start here. Quick wins that provide immediate impact and build momentum.",
                    "skills": tier_output["tier_1"]
                },
                "tier_2": {
                    "name": "Medium ROI + Moderate Difficulty",
                    "description": "Core competency building. Critical for long-term role success.",
                    "skills": tier_output["tier_2"]
                },
                "tier_3": {
                    "name": "Low ROI but Necessary",
                    "description": "Specialized and advanced topics. Required for completeness.",
                    "skills": tier_output["tier_3"]
                }
            },
            "milestones": milestones,
            "timeline": {
                "total_weeks": total_weeks,
                "total_hours": total_hours,
                "target_completion": f"Week {total_weeks}",
                "weekly_hours": weekly_hours,
            }
        }

    # ─── Tier Categorization ──────────────────────────────────────────────────

    def _categorize_into_tiers(
        self,
        gap_skills: List[Dict],
        sequence_order: Dict[str, int]
    ) -> Dict[str, List[Dict]]:
        """
        Assign each gap skill to Tier 1, 2, or 3 based on ROI and difficulty.

        Tier 1: gap_score >= 0.7 AND difficulty <= 2 (High ROI + Low Difficulty)
        Tier 2: gap_score >= 0.4 AND difficulty <= 4 (Medium ROI + Moderate Difficulty)
        Tier 3: Everything else (Low ROI but Necessary)

        Within each tier, order by:
        1. Acquisition sequence (dependency order)
        2. Gap score (highest ROI first)
        """
        tiers = {"tier_1": [], "tier_2": [], "tier_3": []}

        for skill in gap_skills:
            gap_score = skill.get("gap_score", 0)
            difficulty = skill.get("difficulty", 3)

            if gap_score >= 0.7 and difficulty <= 2:
                tiers["tier_1"].append(skill)
            elif gap_score >= 0.4 and difficulty <= 4:
                tiers["tier_2"].append(skill)
            else:
                tiers["tier_3"].append(skill)

        # Sort each tier by (sequence_order, then gap_score desc)
        for tier_name in tiers:
            tiers[tier_name].sort(
                key=lambda s: (
                    sequence_order.get(s["skill_name"], 999),
                    -s.get("gap_score", 0)
                )
            )

        return tiers

    # ─── Build Individual Entry ───────────────────────────────────────────────

    def _build_roadmap_entry(
        self,
        skill: Dict,
        tier: str,
        weekly_hours: int
    ) -> Dict:
        """
        Build a complete roadmap entry for a single skill.

        Args:
            skill: Gap skill dict
            tier: "tier_1", "tier_2", or "tier_3"
            weekly_hours: Learner's weekly study hours

        Returns:
            Roadmap entry dict
        """
        canonical_name = skill.get("skill_name", "Unknown")
        difficulty = skill.get("difficulty", 3)
        learning_hours = skill.get("learning_hours", 80)
        skill_type = skill.get("type", "Technical")
        prerequisites = skill.get("prerequisites", [])
        gap_score = skill.get("gap_score", 0)
        market_demand = skill.get("market_demand", 0.5)
        salary_impact = skill.get("salary_impact", 0.5)

        # Weeks to proficiency (minimum 1 week)
        weeks_to_proficiency = max(math.ceil(learning_hours / max(weekly_hours, 1)), 1)

        # Learning type based on skill category and difficulty
        type_key = (skill_type if skill_type in ["Technical", "Infrastructure", "Soft", "Domain"] else "Technical", difficulty)
        learning_type = LEARNING_TYPE_MAP.get(type_key, "Structured online course + practice projects")

        # Proficiency milestone definition
        if difficulty <= 2:
            proficiency_milestone = f"Build 2-3 small projects demonstrating {canonical_name}"
        elif difficulty <= 3:
            proficiency_milestone = f"Complete a portfolio project and pass a mock assessment in {canonical_name}"
        elif difficulty <= 4:
            proficiency_milestone = f"Deploy a production-level implementation using {canonical_name}"
        else:
            proficiency_milestone = f"Contribute to an open-source project or research using {canonical_name}"

        # Study schedule recommendation
        if learning_hours <= 30:
            schedule = f"{math.ceil(learning_hours / 3)} sessions of ~3 hrs each"
        elif learning_hours <= 100:
            schedule = f"{math.ceil(learning_hours / 5)} sessions of ~5 hrs each, over {weeks_to_proficiency} weeks"
        else:
            schedule = f"Structured program: ~{weekly_hours} hrs/week for {weeks_to_proficiency} weeks"

        return {
            "skill_name": canonical_name,
            "tier": tier,
            "difficulty": difficulty,
            "difficulty_label": self._difficulty_label(difficulty),
            "learning_hours": learning_hours,
            "weeks_to_proficiency": weeks_to_proficiency,
            "learning_type": learning_type,
            "prerequisites": prerequisites,
            "proficiency_milestone": proficiency_milestone,
            "study_schedule": schedule,
            "gap_score": round(gap_score, 4),
            "market_demand": round(market_demand, 3),
            "salary_impact": round(salary_impact, 3),
            "roi_label": self._roi_label(gap_score),
            "start_week": None,   # Filled by _assign_timeline
            "end_week": None,     # Filled by _assign_timeline
        }

    def _difficulty_label(self, difficulty: int) -> str:
        labels = {1: "Beginner", 2: "Easy", 3: "Intermediate", 4: "Advanced", 5: "Expert"}
        return labels.get(difficulty, "Intermediate")

    def _roi_label(self, gap_score: float) -> str:
        if gap_score >= 1.5:
            return "Very High ROI"
        elif gap_score >= 0.8:
            return "High ROI"
        elif gap_score >= 0.4:
            return "Medium ROI"
        elif gap_score >= 0.2:
            return "Low-Medium ROI"
        else:
            return "Low ROI (but required)"

    # ─── Timeline Assignment ───────────────────────────────────────────────────

    def _assign_timeline(self, entries: List[Dict], weekly_hours: int) -> List[Dict]:
        """
        Assign cumulative start_week and end_week to each roadmap entry.
        Respects tier ordering: Tier 1 first, then 2, then 3.
        """
        current_week = 1
        tier_order = ["tier_1", "tier_2", "tier_3"]

        for tier in tier_order:
            tier_entries = [e for e in entries if e["tier"] == tier]
            for entry in tier_entries:
                start = current_week
                end = current_week + entry["weeks_to_proficiency"] - 1
                entry["start_week"] = start
                entry["end_week"] = end
                current_week = end + 1

        return entries

    # ─── Milestones ───────────────────────────────────────────────────────────

    def _generate_milestones(self, tier_output: Dict, total_weeks: int) -> List[Dict]:
        """Generate motivational milestones at key points in the learning journey."""
        milestones = []

        # Tier 1 completion milestone
        tier1_skills = tier_output.get("tier_1", [])
        if tier1_skills:
            last_t1 = tier1_skills[-1]
            milestones.append({
                "week": last_t1.get("end_week", 4),
                "title": "Foundation Complete",
                "description": f"You've mastered {len(tier1_skills)} quick-win skills! "
                              f"You're now ready for more complex topics.",
                "skills_completed": [s["skill_name"] for s in tier1_skills]
            })

        # Tier 2 completion milestone
        tier2_skills = tier_output.get("tier_2", [])
        if tier2_skills:
            last_t2 = tier2_skills[-1]
            milestones.append({
                "week": last_t2.get("end_week", total_weeks // 2),
                "title": "Core Competency Achieved",
                "description": f"Excellent! Core skills acquired. "
                              f"You're now competitive for this role.",
                "skills_completed": [s["skill_name"] for s in tier2_skills]
            })

        # Final completion
        milestones.append({
            "week": total_weeks,
            "title": "Full Readiness",
            "description": "Congratulations! You've completed the learning roadmap. "
                          "You are now fully equipped for this role.",
            "skills_completed": "All gap skills"
        })

        return milestones

    # ─── Quick Wins ───────────────────────────────────────────────────────────

    def _extract_quick_wins(self, tier_output: Dict) -> List[Dict]:
        """
        Extract the first 3 skills across all tiers as motivational quick wins.
        Prioritizes Tier 1 skills, then Tier 2.
        """
        quick_wins = []
        for skill in tier_output.get("tier_1", [])[:3]:
            quick_wins.append({
                "skill_name": skill["skill_name"],
                "weeks": skill["weeks_to_proficiency"],
                "motivation": f"Start with this! Only {skill['weeks_to_proficiency']} week(s) "
                             f"to proficiency and high market demand ({skill['market_demand']:.0%})."
            })

        # Fill up to 3 with tier 2 if tier 1 has < 3
        if len(quick_wins) < 3:
            for skill in tier_output.get("tier_2", [])[:3 - len(quick_wins)]:
                quick_wins.append({
                    "skill_name": skill["skill_name"],
                    "weeks": skill["weeks_to_proficiency"],
                    "motivation": f"High value skill: {skill['roi_label']}. "
                                 f"Adds significant salary impact ({skill['salary_impact']:.0%})."
                })

        return quick_wins
