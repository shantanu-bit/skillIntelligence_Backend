"""
Module G: Course Recommendation Engine
========================================
Hybrid recommendation system combining:

1. Content-Based Filtering:
   - TF-IDF vectors on course descriptions
   - Cosine similarity against skill queries
   - Ranked by relevance, duration, difficulty match

2. Collaborative Filtering:
   - KNN (k=10) on synthetic user-course interactions
   - Similar student profiles weighted by satisfaction

3. Semantic Similarity:
   - SBERT embeddings (all-MiniLM-L6-v2)
   - Matches course descriptions to skill gap needs

Output per skill: Top 3-5 courses with:
{
    "course_name": str,
    "platform": str,
    "level": str,
    "duration_hours": int,
    "semantic_score": float (0-1),
    "rationale": str
}
"""

import json
import logging
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from utils.embedding_service import get_embedding_service
from config.settings import (
    COURSE_CATALOG_PATH,
    KNN_NEIGHBORS,
    MAX_COURSES_PER_SKILL,
    MIN_COURSES_PER_SKILL,
)

logger = logging.getLogger(__name__)


class CourseRecommendationEngine:
    """
    Hybrid course recommendation engine combining content-based,
    collaborative, and semantic similarity approaches.
    """

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.courses = self._load_courses()
        self._tfidf_matrix = None
        self._tfidf_vectorizer = None
        self._knn_model = None
        self._user_item_matrix = None
        self._build_tfidf_index()
        self._build_knn_model()
        logger.info(f"[CourseRecommender] Initialized with {len(self.courses)} courses.")

    # ─── Data Loading ─────────────────────────────────────────────────────────

    def _load_courses(self) -> List[Dict]:
        """Load course catalog from JSON."""
        with open(COURSE_CATALOG_PATH, "r") as f:
            data = json.load(f)
        return data.get("courses", [])

    # ─── 1. Content-Based: TF-IDF Index ──────────────────────────────────────

    def _build_tfidf_index(self):
        """
        Build TF-IDF matrix over all course descriptions.
        Used for content-based cosine similarity matching.
        """
        # Build rich course documents for TF-IDF
        course_docs = []
        for course in self.courses:
            doc = (
                f"{course['name']} {course['description']} "
                f"{' '.join(course['skills_covered'])} "
                f"{course['level']} {course['platform']}"
            )
            course_docs.append(doc)

        self._tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=1,
            stop_words="english",
            sublinear_tf=True  # Use log(tf+1) for better scaling
        )
        self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(course_docs)
        logger.debug(f"[TF-IDF] Matrix shape: {self._tfidf_matrix.shape}")

    def _content_based_score(self, query: str, target_level: Optional[str] = None) -> List[float]:
        """
        Compute TF-IDF cosine similarity scores for all courses vs a query.

        Args:
            query: Skill name or description to match
            target_level: Optional difficulty level filter preference

        Returns:
            List of similarity scores (one per course)
        """
        query_vec = self._tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix)[0]

        # Apply difficulty-level preference boost
        if target_level:
            level_order = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
            target_order = level_order.get(target_level, 2)
            for i, course in enumerate(self.courses):
                course_order = level_order.get(course.get("level", "Intermediate"), 2)
                # Boost courses at the target level, slight penalty for others
                if course_order == target_order:
                    scores[i] *= 1.15
                elif abs(course_order - target_order) == 1:
                    scores[i] *= 0.95
                else:
                    scores[i] *= 0.80

        return scores.tolist()

    # ─── 2. Collaborative Filtering: KNN ─────────────────────────────────────

    def _build_knn_model(self):
        """
        Build KNN model on a synthetic user-course interaction matrix.
        In production, this would use real user enrollment + satisfaction data.

        For this implementation, we generate a plausible synthetic matrix
        based on course ratings, enrollments, and skill overlap patterns.
        """
        n_users = 200
        n_courses = len(self.courses)

        # Synthetic user-course interaction matrix (users × courses)
        # Scores represent satisfaction (0=not taken, 1-5 = rating)
        rng = np.random.default_rng(seed=42)
        interaction_matrix = np.zeros((n_users, n_courses))

        # Simulate user enrollment patterns based on course characteristics
        for user_idx in range(n_users):
            # Assign each user a random skill level preference (1=beginner, 2=inter, 3=adv)
            user_level_pref = rng.integers(1, 4)
            # Assign domain preference
            user_domain = rng.choice(["python_ml", "web_dev", "data_eng", "infra"])

            for course_idx, course in enumerate(self.courses):
                # Calculate enrollment probability based on user-course affinity
                course_level_num = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}.get(
                    course.get("level", "Intermediate"), 2
                )
                level_match = 1.0 - (abs(user_level_pref - course_level_num) / 3.0)

                # Domain affinity
                course_skills = " ".join(course.get("skills_covered", [])).lower()
                domain_match = 0.3
                if user_domain == "python_ml" and any(s in course_skills for s in ["python", "machine", "deep"]):
                    domain_match = 0.8
                elif user_domain == "web_dev" and any(s in course_skills for s in ["react", "javascript", "django"]):
                    domain_match = 0.8
                elif user_domain == "data_eng" and any(s in course_skills for s in ["spark", "sql", "data engineering"]):
                    domain_match = 0.8
                elif user_domain == "infra" and any(s in course_skills for s in ["docker", "kubernetes", "aws"]):
                    domain_match = 0.8

                enrollment_prob = level_match * domain_match * 0.7
                if rng.random() < enrollment_prob:
                    # Assign satisfaction score based on course rating + noise
                    base_rating = course.get("rating", 4.5) / 5.0  # Normalize to 0-1
                    noise = rng.normal(0, 0.1)
                    satisfaction = float(np.clip(base_rating + noise, 0.0, 1.0))
                    interaction_matrix[user_idx, course_idx] = satisfaction

        self._user_item_matrix = interaction_matrix

        # Fit KNN on user vectors (user-based collaborative filtering)
        self._knn_model = NearestNeighbors(
            n_neighbors=KNN_NEIGHBORS,
            metric="cosine",
            algorithm="brute"
        )
        self._knn_model.fit(interaction_matrix)
        logger.debug(f"[KNN] Model fitted on {n_users}x{n_courses} interaction matrix.")

    def _collaborative_score(self, query_skills: List[str]) -> List[float]:
        """
        Collaborative filtering score via KNN.
        Simulate a "target user" whose interests match the query skills,
        find k=10 similar users, and aggregate their course preferences.

        Args:
            query_skills: List of skill names to build target user profile

        Returns:
            Aggregated collaborative score per course
        """
        n_courses = len(self.courses)

        # Build a synthetic target user vector based on query skills
        target_user = np.zeros(n_courses)
        for course_idx, course in enumerate(self.courses):
            skills_lower = [s.lower() for s in course.get("skills_covered", [])]
            query_lower = [s.lower() for s in query_skills]
            overlap = sum(1 for s in query_lower if any(s in sl or sl in s for sl in skills_lower))
            if overlap > 0:
                base_score = course.get("rating", 4.0) / 5.0
                target_user[course_idx] = base_score * min(overlap / len(query_skills), 1.0)

        # Find k=10 nearest neighbors
        target_reshaped = target_user.reshape(1, -1)
        distances, indices = self._knn_model.kneighbors(target_reshaped)

        # Aggregate neighbor preferences (weighted by 1 - distance = similarity)
        collaborative_scores = np.zeros(n_courses)
        neighbor_weights = 1.0 - distances[0]  # Convert distance to similarity

        for neighbor_idx, weight in zip(indices[0], neighbor_weights):
            collaborative_scores += self._user_item_matrix[neighbor_idx] * weight

        # Normalize to 0-1
        max_score = collaborative_scores.max()
        if max_score > 0:
            collaborative_scores = collaborative_scores / max_score

        return collaborative_scores.tolist()

    # ─── 3. Semantic Similarity: SBERT ───────────────────────────────────────

    def _semantic_score(self, skill_name: str, skill_description: str) -> List[float]:
        """
        SBERT semantic similarity between skill need and course descriptions.

        Args:
            skill_name: Name of the skill gap
            skill_description: Description of what the skill involves

        Returns:
            Semantic similarity score per course (0-1)
        """
        query = f"{skill_name}. {skill_description}"
        query_embedding = self.embedding_service.encode([query], normalize=True)[0]

        _, course_embeddings = self.embedding_service.get_course_embeddings()

        if len(course_embeddings) == 0:
            return [0.0] * len(self.courses)

        similarities = self.embedding_service.cosine_similarity(query_embedding, course_embeddings)
        return similarities.tolist()

    # ─── Hybrid Aggregation ───────────────────────────────────────────────────

    def recommend(
        self,
        skill_name: str,
        skill_metadata: Optional[Dict] = None,
        top_n: int = MAX_COURSES_PER_SKILL
    ) -> List[Dict]:
        """
        Generate hybrid course recommendations for a single skill gap.

        Combines:
        - Content-based (TF-IDF): weight 0.35
        - Collaborative (KNN): weight 0.25
        - Semantic (SBERT): weight 0.40

        Args:
            skill_name: Name of the skill to find courses for
            skill_metadata: Optional metadata dict (difficulty, type, etc.)
            top_n: Max courses to return (default 5, min 3)

        Returns:
            List of course recommendation dicts
        """
        if not self.courses:
            return []

        top_n = max(top_n, MIN_COURSES_PER_SKILL)
        skill_metadata = skill_metadata or {}

        # Infer target difficulty from skill metadata
        difficulty = skill_metadata.get("difficulty", 3)
        target_level = {1: "Beginner", 2: "Beginner", 3: "Intermediate", 4: "Advanced", 5: "Advanced"}.get(difficulty, "Intermediate")

        # Build query for content-based and semantic
        skill_desc = f"{skill_name} programming technology framework tool"
        if skill_metadata.get("type"):
            skill_desc += f" {skill_metadata['type']}"

        # Compute individual scores
        cb_scores = self._content_based_score(
            query=f"{skill_name} {skill_desc}",
            target_level=target_level
        )
        collab_scores = self._collaborative_score(query_skills=[skill_name])
        semantic_scores = self._semantic_score(skill_name, skill_desc)

        # Hybrid fusion: weighted combination
        WEIGHT_CONTENT = 0.35
        WEIGHT_COLLAB = 0.25
        WEIGHT_SEMANTIC = 0.40

        hybrid_scores = []
        for i in range(len(self.courses)):
            score = (
                WEIGHT_CONTENT * cb_scores[i] +
                WEIGHT_COLLAB * collab_scores[i] +
                WEIGHT_SEMANTIC * semantic_scores[i]
            )
            hybrid_scores.append(score)

        # Get top-N courses
        ranked_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)
        top_indices = ranked_indices[:top_n]

        recommendations = []
        for idx in top_indices:
            course = self.courses[idx]
            hybrid_score = hybrid_scores[idx]

            if hybrid_score < 0.01:
                continue  # Skip irrelevant courses

            rec = {
                "course_name": course["name"],
                "platform": course["platform"],
                "level": course["level"],
                "duration_hours": course["duration_hours"],
                "semantic_score": round(float(semantic_scores[idx]), 4),
                "content_score": round(float(cb_scores[idx]), 4),
                "collaborative_score": round(float(collab_scores[idx]), 4),
                "hybrid_score": round(float(hybrid_score), 4),
                "rating": course.get("rating", 4.5),
                "enrollments": course.get("enrollments", 0),
                "price_usd": course.get("price_usd", 0),
                "certificate": course.get("certificate", False),
                "url": course.get("url", ""),
                "skills_covered": course.get("skills_covered", []),
                "rationale": self._generate_rationale(
                    skill_name=skill_name,
                    course=course,
                    content_score=cb_scores[idx],
                    collab_score=collab_scores[idx],
                    semantic_score=float(semantic_scores[idx]),
                    target_level=target_level
                )
            }
            recommendations.append(rec)

        logger.debug(f"[CourseRecommender] {len(recommendations)} recommendations for '{skill_name}'.")
        return recommendations

    def recommend_all_gaps(
        self,
        gap_analysis: Dict,
        top_n: int = MAX_COURSES_PER_SKILL
    ) -> Dict[str, List[Dict]]:
        """
        Generate recommendations for all skill gaps from gap analysis.

        Args:
            gap_analysis: Output from SkillGapAnalysisEngine
            top_n: Max courses per skill

        Returns:
            Dict: {skill_name -> [course recommendations]}
        """
        skill_gaps = gap_analysis.get("skill_gaps", [])
        recommendations = {}

        for gap_skill in skill_gaps:
            skill_name = gap_skill.get("skill_name", "")
            if not skill_name:
                continue

            recs = self.recommend(
                skill_name=skill_name,
                skill_metadata=gap_skill,
                top_n=top_n
            )
            recommendations[skill_name] = recs

        logger.info(
            f"[CourseRecommender] Generated recommendations for "
            f"{len(recommendations)} skills."
        )
        return recommendations

    # ─── Rationale Generation ─────────────────────────────────────────────────

    def _generate_rationale(
        self,
        skill_name: str,
        course: Dict,
        content_score: float,
        collab_score: float,
        semantic_score: float,
        target_level: str
    ) -> str:
        """Generate a human-readable explanation for the recommendation."""
        reasons = []

        # Best match indicator
        if semantic_score >= 0.7:
            reasons.append(f"semantically highly aligned with {skill_name}")
        elif semantic_score >= 0.5:
            reasons.append(f"good semantic match for {skill_name}")

        # Content relevance
        if content_score >= 0.3:
            reasons.append(f"covers relevant keywords and technology stack")

        # Collaborative signal
        if collab_score >= 0.5:
            reasons.append("popular among learners with similar profiles")

        # Course quality signals
        rating = course.get("rating", 4.0)
        enrollments = course.get("enrollments", 0)
        if rating >= 4.7:
            reasons.append(f"top-rated ({rating}/5.0)")
        if enrollments >= 500000:
            reasons.append(f"highly popular ({enrollments:,} learners)")

        # Level match
        if course.get("level") == target_level:
            reasons.append(f"matches your target level ({target_level})")

        # Certificate
        if course.get("certificate"):
            reasons.append("includes verifiable certificate")

        # Price
        if course.get("price_usd", 99) == 0:
            reasons.append("free course")
        elif course.get("price_usd", 99) <= 20:
            reasons.append("affordable pricing")

        if not reasons:
            reasons = [f"relevant content coverage for {skill_name}"]

        return f"Recommended because: {'; '.join(reasons[:4])}."
