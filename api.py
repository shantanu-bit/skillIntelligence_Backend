"""
api.py — Flask REST API for the AI Skill Intelligence Platform
==============================================================
Wraps all 7 pipeline engines behind HTTP endpoints.
The React UI calls these instead of the Anthropic API.

Run:
    pip install flask flask-cors
    python api.py

Server starts on: http://localhost:5000

Endpoints:
    POST /api/analyze          — Full 7-step pipeline (text + file upload)
    POST /api/extract-resume   — Module A+B: extract + mine resume skills
    POST /api/mine-jd          — Module D: mine JD skills
    POST /api/gap-analysis     — Module E: compute skill gaps
    POST /api/roadmap          — Module F: generate learning roadmap
    POST /api/courses          — Module G: course recommendations
    GET  /api/health           — Health check
    GET  /api/taxonomy         — Return full skill taxonomy
"""

import os
import sys
import json
import time
import logging
import tempfile
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

# ─── Path Setup ───────────────────────────────────────────────────────────────
# Ensures all skill_intelligence_platform modules are importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("SkillIQ.API")

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Allow all origins in dev (lock this down in production)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ─── Platform Singleton (warm up engines once on startup) ────────────────────
_platform = None

def get_platform():
    """Lazy-load the platform singleton (engines load on first request)."""
    global _platform
    if _platform is None:
        from main import SkillIntelligencePlatform
        _platform = SkillIntelligencePlatform()
        logger.info("Platform initialized.")
    return _platform


# ─── Error Helper ─────────────────────────────────────────────────────────────
def err(message: str, status: int = 400):
    return jsonify({"error": message}), status


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    """Quick liveness check — also warms up the platform."""
    platform = get_platform()
    return jsonify({
        "status": "ok",
        "platform": "AI Skill Intelligence Platform v1.0",
        "engines": {
            "text_extraction": True,
            "skill_extraction": True,
            "jd_mining": True,
            "gap_analysis": True,
            "roadmap": True,
            "courses": True,
        }
    })


# ─── /api/analyze ─────────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Full 7-step pipeline in one call.

    Accepts multipart/form-data OR application/json.

    Form-data fields:
        resume_text   (str) — raw resume text
        resume_file   (file) — PDF or DOCX upload (alternative to resume_text)
        jd_text       (str) — job description text
        weekly_hours  (int, optional, default=15)

    JSON body:
        { "resume_text": "...", "jd_text": "...", "weekly_hours": 15 }

    Returns: Full analysis JSON (same schema as main.py output)
    """
    t0 = time.time()

    # ── Parse inputs ──────────────────────────────────────────────────────────
    if request.content_type and "multipart" in request.content_type:
        resume_text = request.form.get("resume_text", "").strip()
        jd_text = request.form.get("jd_text", "").strip()
        weekly_hours = int(request.form.get("weekly_hours", 15))
        resume_file = request.files.get("resume_file")
    else:
        body = request.get_json(silent=True) or {}
        resume_text = body.get("resume_text", "").strip()
        jd_text = body.get("jd_text", "").strip()
        weekly_hours = int(body.get("weekly_hours", 15))
        resume_file = None

    if not jd_text:
        return err("jd_text is required.")

    # ── Handle file upload ────────────────────────────────────────────────────
    resume_path = None
    tmp_file = None

    if resume_file and resume_file.filename:
        ext = Path(resume_file.filename).suffix.lower()
        if ext not in [".pdf", ".docx"]:
            return err("resume_file must be a .pdf or .docx file.")
        tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        resume_file.save(tmp_file.name)
        resume_path = tmp_file.name
        logger.info(f"Uploaded resume saved to: {resume_path}")

    if not resume_path and not resume_text:
        return err("Provide either resume_text or a resume_file upload.")

    try:
        platform = get_platform()
        result = platform.analyze(
            resume_path=resume_path,
            resume_text=resume_text if not resume_path else None,
            jd_text=jd_text,
            weekly_hours=weekly_hours,
        )
        result["metadata"]["api_processing_seconds"] = round(time.time() - t0, 2)
        logger.info(f"Full analysis completed in {result['metadata']['api_processing_seconds']}s")
        return jsonify(result)

    except Exception as e:
        logger.exception("Error in /api/analyze")
        return err(f"Analysis failed: {str(e)}", 500)

    finally:
        # Clean up temp file
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


# ─── /api/extract-resume ──────────────────────────────────────────────────────
@app.route("/api/extract-resume", methods=["POST"])
def extract_resume():
    """
    Modules A + B: text extraction + hybrid skill extraction.

    JSON body:
        { "resume_text": "...", "mode": "resume" }
        OR multipart with resume_file

    Returns:
        { "sections": {...}, "skills": [...], "skills_found": N }
    """
    if request.content_type and "multipart" in request.content_type:
        resume_text = request.form.get("resume_text", "").strip()
        resume_file = request.files.get("resume_file")
        mode = request.form.get("mode", "resume")
    else:
        body = request.get_json(silent=True) or {}
        resume_text = body.get("resume_text", "").strip()
        resume_file = None
        mode = body.get("mode", "resume")

    resume_path = None
    tmp_file = None

    if resume_file and resume_file.filename:
        ext = Path(resume_file.filename).suffix.lower()
        tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        resume_file.save(tmp_file.name)
        resume_path = tmp_file.name

    if not resume_path and not resume_text:
        return err("Provide resume_text or resume_file.")

    try:
        platform = get_platform()

        # Step 1: Extract text sections
        if resume_path:
            sections = platform.text_engine.extract(resume_path)
        else:
            sections = platform.text_engine.extract_from_text(resume_text)

        full_text = sections.get("full_text", "")

        # Step 2: Extract skills
        skills = platform.skill_engine.extract_skills(text=full_text, mode=mode)

        return jsonify({
            "sections": {k: v for k, v in sections.items() if k != "full_text"},
            "sections_detected": {k: bool(v) for k, v in sections.items() if k != "full_text"},
            "skills": skills,
            "skills_found": len(skills),
            "full_text_length": len(full_text),
        })

    except Exception as e:
        logger.exception("Error in /api/extract-resume")
        return err(str(e), 500)

    finally:
        if tmp_file and os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)


# ─── /api/mine-jd ─────────────────────────────────────────────────────────────
@app.route("/api/mine-jd", methods=["POST"])
def mine_jd():
    """
    Module D: Job Description skill mining.

    JSON body: { "jd_text": "..." }

    Returns:
        { "required_skills": [...], "detected_industry": "...", "total_skills_found": N }
    """
    body = request.get_json(silent=True) or {}
    jd_text = body.get("jd_text", "").strip()
    if not jd_text:
        return err("jd_text is required.")

    try:
        platform = get_platform()
        result = platform.jd_miner.mine_skills(jd_text)
        return jsonify(result)
    except Exception as e:
        logger.exception("Error in /api/mine-jd")
        return err(str(e), 500)


# ─── /api/gap-analysis ────────────────────────────────────────────────────────
@app.route("/api/gap-analysis", methods=["POST"])
def gap_analysis():
    """
    Module E: Skill Gap Analysis.

    JSON body:
        { "resume_skills": [...], "jd_skills": [...] }

    Returns:
        { "summary": {...}, "matched_skills": [...], "skill_gaps": [...], "acquisition_sequence": [...] }
    """
    body = request.get_json(silent=True) or {}
    resume_skills = body.get("resume_skills", [])
    jd_skills = body.get("jd_skills", [])

    if not resume_skills:
        return err("resume_skills array is required.")
    if not jd_skills:
        return err("jd_skills array is required.")

    try:
        platform = get_platform()
        result = platform.gap_engine.analyze(
            resume_skills=resume_skills,
            jd_skills=jd_skills
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("Error in /api/gap-analysis")
        return err(str(e), 500)


# ─── /api/roadmap ─────────────────────────────────────────────────────────────
@app.route("/api/roadmap", methods=["POST"])
def roadmap():
    """
    Module F: Learning Roadmap Generator.

    JSON body:
        { "gap_analysis": {...}, "weekly_hours": 15 }

    Returns:
        Full roadmap with tiers, milestones, quick wins, timeline.
    """
    body = request.get_json(silent=True) or {}
    gap_data = body.get("gap_analysis", {})
    weekly_hours = int(body.get("weekly_hours", 15))

    if not gap_data:
        return err("gap_analysis object is required.")

    try:
        platform = get_platform()
        result = platform.roadmap_generator.generate(
            gap_analysis=gap_data,
            weekly_hours=weekly_hours
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("Error in /api/roadmap")
        return err(str(e), 500)


# ─── /api/courses ─────────────────────────────────────────────────────────────
@app.route("/api/courses", methods=["POST"])
def courses():
    """
    Module G: Hybrid Course Recommendation Engine.

    JSON body:
        { "gap_analysis": {...}, "top_n": 5 }

    Returns:
        { "SkillName": [ { course_name, platform, semantic_score, ... } ] }
    """
    body = request.get_json(silent=True) or {}
    gap_data = body.get("gap_analysis", {})
    top_n = int(body.get("top_n", 5))

    if not gap_data:
        return err("gap_analysis object is required.")

    try:
        platform = get_platform()
        result = platform.course_recommender.recommend_all_gaps(
            gap_analysis=gap_data,
            top_n=top_n
        )
        return jsonify(result)
    except Exception as e:
        logger.exception("Error in /api/courses")
        return err(str(e), 500)


# ─── /api/taxonomy ────────────────────────────────────────────────────────────
@app.route("/api/taxonomy", methods=["GET"])
def taxonomy():
    """
    Module C: Return the full skill taxonomy with market stats.
    Useful for browsing available skills and their metadata.
    """
    try:
        platform = get_platform()
        return jsonify({
            "skills": platform.taxonomy.get_all_skills(),
            "stats": platform.taxonomy.get_market_stats(),
        })
    except Exception as e:
        logger.exception("Error in /api/taxonomy")
        return err(str(e), 500)


@app.route("/api/taxonomy/search", methods=["GET"])
def taxonomy_search():
    """
    Semantic search over the taxonomy.
    Query param: ?q=natural+language+processing&top_k=5
    """
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 10))

    if not query:
        return err("q query parameter is required.")

    try:
        platform = get_platform()
        results = platform.taxonomy.semantic_search(query, top_k=top_k)
        return jsonify({"query": query, "results": results})
    except Exception as e:
        logger.exception("Error in /api/taxonomy/search")
        return err(str(e), 500)


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SkillIQ API Server")
    print("  http://localhost:5000")
    print("=" * 60)
    print("\n  Endpoints:")
    print("    GET  /api/health")
    print("    GET  /api/taxonomy")
    print("    GET  /api/taxonomy/search?q=machine+learning")
    print("    POST /api/analyze          ← Full pipeline")
    print("    POST /api/extract-resume")
    print("    POST /api/mine-jd")
    print("    POST /api/gap-analysis")
    print("    POST /api/roadmap")
    print("    POST /api/courses")
    print("\n  Press CTRL+C to stop.\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,         # Set to False in production
        use_reloader=False, # Prevents double-loading the heavy ML models
    )
