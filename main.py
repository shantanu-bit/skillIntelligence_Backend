"""
AI Skill Intelligence Platform - Main Pipeline
================================================
Orchestrates all 7 modules in the correct sequence:

1. Text Extraction Engine          → Structured resume text
2. Hybrid Skill Extraction Engine  → Resume skills with strength scores
3. Job Description Skill Mining    → JD required skills
4. Skill Taxonomy System           → Skill metadata enrichment
5. Skill Gap Analysis Engine       → Gap scores + acquisition sequence
6. Learning Roadmap Generator      → Tiered roadmap
7. Course Recommendation Engine    → Course matches per gap skill

Usage:
    platform = SkillIntelligencePlatform()
    result = platform.analyze(resume_path="resume.pdf", jd_text="...")
    result = platform.analyze(resume_text="...", jd_text="...")
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SkillIntelligencePlatform")


class SkillIntelligencePlatform:
    """
    Main orchestrator for the AI Skill Intelligence Platform.
    Lazily initializes all engines on first use for memory efficiency.
    """

    def __init__(self, weekly_hours: int = 10):
        """
        Args:
            weekly_hours: Learner's weekly study hours (affects roadmap timeline)
        """
        self.weekly_hours = weekly_hours

        # Lazy-loaded engines
        self._text_engine = None
        self._skill_engine = None
        self._taxonomy_system = None
        self._jd_miner = None
        self._gap_engine = None
        self._roadmap_generator = None
        self._course_recommender = None

        logger.info("=== AI Skill Intelligence Platform Initialized ===")

    # ─── Engine Accessors (Lazy Init) ─────────────────────────────────────────

    @property
    def text_engine(self):
        if self._text_engine is None:
            from engines.text_extraction_engine import TextExtractionEngine
            self._text_engine = TextExtractionEngine()
        return self._text_engine

    @property
    def skill_engine(self):
        if self._skill_engine is None:
            from engines.hybrid_skill_extraction_engine import HybridSkillExtractionEngine
            self._skill_engine = HybridSkillExtractionEngine()
        return self._skill_engine

    @property
    def taxonomy(self):
        if self._taxonomy_system is None:
            from engines.skill_taxonomy_system import SkillTaxonomySystem
            self._taxonomy_system = SkillTaxonomySystem()
        return self._taxonomy_system

    @property
    def jd_miner(self):
        if self._jd_miner is None:
            from engines.jd_skill_miner import JobDescriptionSkillMiner
            self._jd_miner = JobDescriptionSkillMiner()
        return self._jd_miner

    @property
    def gap_engine(self):
        if self._gap_engine is None:
            from engines.skill_gap_analysis_engine import SkillGapAnalysisEngine
            self._gap_engine = SkillGapAnalysisEngine()
        return self._gap_engine

    @property
    def roadmap_generator(self):
        if self._roadmap_generator is None:
            from engines.learning_roadmap_generator import LearningRoadmapGenerator
            self._roadmap_generator = LearningRoadmapGenerator()
        return self._roadmap_generator

    @property
    def course_recommender(self):
        if self._course_recommender is None:
            from engines.course_recommendation_engine import CourseRecommendationEngine
            self._course_recommender = CourseRecommendationEngine()
        return self._course_recommender

    # ─── Main Analysis Interface ──────────────────────────────────────────────

    def analyze(
        self,
        jd_text: str,
        resume_path: Optional[str] = None,
        resume_text: Optional[str] = None,
        weekly_hours: Optional[int] = None,
    ) -> Dict:
        """
        Run the full 7-module pipeline.

        Args:
            jd_text: Raw job description text
            resume_path: Path to resume PDF or DOCX file (mutually exclusive with resume_text)
            resume_text: Raw resume text string (mutually exclusive with resume_path)
            weekly_hours: Learner's weekly study commitment (overrides constructor value)

        Returns:
            Complete analysis result dict
        """
        if not resume_path and not resume_text:
            raise ValueError("Provide either resume_path or resume_text.")
        if not jd_text:
            raise ValueError("Job description text (jd_text) is required.")

        weekly_hours = weekly_hours or self.weekly_hours
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("STEP 1: Text Extraction")
        logger.info("=" * 60)

        # ── Step 1: Extract Resume Text ────────────────────────────────────────
        if resume_path:
            resume_sections = self.text_engine.extract(resume_path)
        else:
            resume_sections = self.text_engine.extract_from_text(resume_text)

        resume_full_text = resume_sections.get("full_text", "")
        logger.info(f"Extracted {len(resume_full_text)} chars from resume.")

        logger.info("=" * 60)
        logger.info("STEP 2: Hybrid Skill Extraction (Resume)")
        logger.info("=" * 60)

        # ── Step 2: Extract Skills from Resume ────────────────────────────────
        resume_skills = self.skill_engine.extract_skills(
            text=resume_full_text,
            mode="resume"
        )
        logger.info(f"Extracted {len(resume_skills)} skills from resume.")

        logger.info("=" * 60)
        logger.info("STEP 3: Job Description Skill Mining")
        logger.info("=" * 60)

        # ── Step 3: Mine JD Skills ────────────────────────────────────────────
        jd_result = self.jd_miner.mine_skills(jd_text)
        jd_skills = jd_result.get("required_skills", [])
        logger.info(
            f"Mined {len(jd_skills)} required skills from JD "
            f"(industry: {jd_result.get('detected_industry')})"
        )

        logger.info("=" * 60)
        logger.info("STEP 4: Skill Gap Analysis")
        logger.info("=" * 60)

        # ── Step 4: Gap Analysis ──────────────────────────────────────────────
        gap_analysis = self.gap_engine.analyze(
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )
        summary = gap_analysis.get("summary", {})
        logger.info(
            f"Gap analysis: {summary.get('match_percentage')}% match, "
            f"{summary.get('total_gaps')} gaps, "
            f"profile strength: {summary.get('profile_strength')}"
        )

        logger.info("=" * 60)
        logger.info("STEP 5: Learning Roadmap Generation")
        logger.info("=" * 60)

        # ── Step 5: Learning Roadmap ───────────────────────────────────────────
        roadmap = self.roadmap_generator.generate(
            gap_analysis=gap_analysis,
            weekly_hours=weekly_hours
        )
        logger.info(
            f"Roadmap: {roadmap.get('roadmap_summary', {}).get('total_weeks_estimated')} weeks, "
            f"{roadmap.get('roadmap_summary', {}).get('total_hours_required')} total hours."
        )

        logger.info("=" * 60)
        logger.info("STEP 6: Course Recommendations")
        logger.info("=" * 60)

        # ── Step 6: Course Recommendations ────────────────────────────────────
        course_recommendations = self.course_recommender.recommend_all_gaps(
            gap_analysis=gap_analysis,
            top_n=5
        )
        logger.info(
            f"Generated course recommendations for "
            f"{len(course_recommendations)} skill gaps."
        )

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"=== Analysis complete in {elapsed}s ===")

        # ── Build Final Output ─────────────────────────────────────────────────
        return {
            "metadata": {
                "platform": "AI Skill Intelligence Platform v1.0",
                "processing_time_seconds": elapsed,
                "weekly_hours_assumed": weekly_hours,
            },
            "resume_analysis": {
                "sections_detected": {
                    k: bool(v) for k, v in resume_sections.items()
                    if k != "full_text"
                },
                "skills_found": len(resume_skills),
                "top_skills": resume_skills[:10],
                "all_skills": resume_skills,
            },
            "jd_analysis": {
                "skills_required": len(jd_skills),
                "detected_industry": jd_result.get("detected_industry"),
                "mining_stats": jd_result.get("jd_summary"),
                "required_skills": jd_skills,
            },
            "gap_analysis": gap_analysis,
            "roadmap": roadmap,
            "course_recommendations": course_recommendations,
        }

    def analyze_resume_only(
        self,
        resume_path: Optional[str] = None,
        resume_text: Optional[str] = None
    ) -> Dict:
        """
        Extract and analyze skills from a resume without a JD comparison.
        Useful for standalone resume analysis.
        """
        if not resume_path and not resume_text:
            raise ValueError("Provide either resume_path or resume_text.")

        if resume_path:
            sections = self.text_engine.extract(resume_path)
        else:
            sections = self.text_engine.extract_from_text(resume_text)

        skills = self.skill_engine.extract_skills(
            text=sections.get("full_text", ""),
            mode="resume"
        )

        return {
            "sections": sections,
            "skills": skills,
            "taxonomy_insights": self.taxonomy.get_market_stats()
        }

    def analyze_jd_only(self, jd_text: str) -> Dict:
        """
        Mine skills from a job description standalone.
        """
        return self.jd_miner.mine_skills(jd_text)

    def save_result(self, result: Dict, output_path: str):
        """
        Save analysis result to JSON file.

        Args:
            result: Analysis result dict from analyze()
            output_path: Path to write JSON output
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Result saved to: {output_path}")


# ─── Demo / Entry Point ───────────────────────────────────────────────────────

def main():
    """
    Demo run with example resume and job description.
    """
    print("\n" + "=" * 70)
    print("AI SKILL INTELLIGENCE PLATFORM — DEMO RUN")
    print("=" * 70 + "\n")

    # ─── Example Resume Text ──────────────────────────────────────────────────
    EXAMPLE_RESUME = """
    John Smith
    john.smith@email.com | +1 (555) 123-4567 | LinkedIn: linkedin.com/in/johnsmith

    SUMMARY
    Data Engineer with 3 years of experience building scalable data pipelines
    and analytics infrastructure. Proficient in Python and SQL. Looking to transition
    into Machine Learning Engineering.

    EXPERIENCE
    Senior Data Analyst | TechCorp Inc. | 2021 - Present
    - Built and maintained ETL pipelines using Python and SQL for 10M+ daily records
    - Developed automated reporting dashboards using Pandas and data visualization tools
    - Collaborated with cross-functional teams to deliver data-driven business insights
    - Optimized distributed data pipelines reducing processing time by 40%

    Junior Developer | StartupXYZ | 2020 - 2021
    - Developed REST APIs using Python Flask
    - Used Git for version control in an Agile team environment
    - Wrote SQL queries for PostgreSQL database management

    EDUCATION
    B.S. Computer Science | State University | 2020
    Relevant coursework: Data Structures, Algorithms, Databases, Statistics

    SKILLS
    Programming: Python, SQL, JavaScript (basic)
    Tools: Git, Docker (basic), Linux, Pandas, NumPy
    Databases: PostgreSQL, MySQL
    Soft Skills: Communication, Problem Solving, Teamwork

    CERTIFICATIONS
    AWS Cloud Practitioner (2022)
    """

    # ─── Example Job Description ──────────────────────────────────────────────
    EXAMPLE_JD = """
    Machine Learning Engineer | DataDriven AI | San Francisco, CA

    We are looking for a skilled Machine Learning Engineer to join our AI team.

    Required Skills:
    - Strong proficiency in Python and Machine Learning frameworks
    - Experience with Deep Learning and PyTorch or TensorFlow
    - Knowledge of MLOps practices including Docker, Kubernetes, and CI/CD
    - Proficiency in SQL and data processing at scale (Apache Spark preferred)
    - Understanding of NLP and Natural Language Processing techniques
    - Experience with AWS cloud services
    - Strong communication and leadership skills
    - Git version control and collaborative development

    Preferred:
    - Experience with Kubernetes orchestration
    - Knowledge of distributed systems
    - Data Science background
    - Experience with React for building ML model interfaces

    About Us:
    DataDriven AI is a fast-growing fintech startup building next-generation
    credit scoring models using machine learning and alternative data sources.
    """

    # ─── Initialize Platform ──────────────────────────────────────────────────
    platform = SkillIntelligencePlatform(weekly_hours=15)

    # ─── Run Full Analysis ────────────────────────────────────────────────────
    result = platform.analyze(
        resume_text=EXAMPLE_RESUME,
        jd_text=EXAMPLE_JD,
        weekly_hours=15
    )

    # ─── Save Output ──────────────────────────────────────────────────────────
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "outputs",
        "analysis_result.json"
    )
    platform.save_result(result, output_path)

    # ─── Print Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS SUMMARY")
    print("=" * 70)

    summary = result["gap_analysis"].get("summary", {})
    print(f"\n📋 Resume Skills Found:    {result['resume_analysis']['skills_found']}")
    print(f"📋 JD Skills Required:     {result['jd_analysis']['skills_required']}")
    print(f"✅ Skills Matched:         {summary.get('total_matched')}")
    print(f"❌ Skills Gap:             {summary.get('total_gaps')}")
    print(f"📊 Match Percentage:       {summary.get('match_percentage')}%")
    print(f"💪 Profile Strength:       {summary.get('profile_strength')}")
    print(f"⚠️  Avg Gap Severity:       {summary.get('weighted_avg_severity'):.4f}")

    print("\n--- TOP RESUME SKILLS ---")
    for skill in result["resume_analysis"]["top_skills"][:7]:
        print(f"  ✓ {skill['skill_name']} "
              f"(strength: {skill['strength_score']:.2f}, "
              f"method: {skill['extraction_method']})")

    print("\n--- SKILL GAPS (ordered by priority) ---")
    for gap in result["gap_analysis"].get("skill_gaps", [])[:8]:
        print(f"  ✗ {gap['skill_name']} "
              f"(gap_score: {gap['gap_score']:.4f}, "
              f"difficulty: {gap['difficulty']}/5, "
              f"hours: {gap['learning_hours']}h)")

    print("\n--- ACQUISITION SEQUENCE (dependency-aware) ---")
    for step in result["gap_analysis"].get("acquisition_sequence", [])[:8]:
        print(f"  Step {step['step']}: {step['skill_name']} "
              f"— {step.get('reason', '')[:60]}...")

    roadmap_summary = result["roadmap"].get("roadmap_summary", {})
    print(f"\n--- LEARNING ROADMAP ---")
    print(f"  Total Skills:  {roadmap_summary.get('total_skills_to_learn')}")
    print(f"  Total Hours:   {roadmap_summary.get('total_hours_required')}")
    print(f"  Duration:      {roadmap_summary.get('total_weeks_estimated')} weeks "
          f"({15} hrs/week)")

    print("\n--- QUICK WINS ---")
    for qw in result["roadmap"].get("quick_wins", []):
        print(f"  🚀 {qw['skill_name']} ({qw['weeks']} week(s)) — {qw['motivation'][:60]}...")

    print("\n--- COURSE RECOMMENDATIONS (sample) ---")
    course_recs = result["course_recommendations"]
    shown = 0
    for skill_name, courses in course_recs.items():
        if shown >= 3:
            break
        print(f"\n  Skill: {skill_name}")
        for c in courses[:2]:
            print(f"    📚 {c['course_name']}")
            print(f"       Platform: {c['platform']} | Level: {c['level']} | "
                  f"{c['duration_hours']}h | Semantic: {c['semantic_score']:.3f}")
            print(f"       {c['rationale'][:80]}...")
        shown += 1

    print(f"\n\n✅ Full results saved to: {output_path}")
    print("=" * 70 + "\n")
    return result


if __name__ == "__main__":
    main()
