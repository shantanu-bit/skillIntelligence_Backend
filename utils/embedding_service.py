"""
Shared Embedding Service
Provides cached, reusable Sentence-BERT embeddings across all modules.
Model: all-MiniLM-L6-v2
"""

import os
import json
import numpy as np
import logging
from typing import List, Optional, Dict
from utils.compat import SentenceTransformer
from config.settings import SBERT_MODEL, TAXONOMY_PATH, EMBEDDING_CACHE_PATH, COURSE_CATALOG_PATH, COURSE_EMBEDDING_CACHE

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton-style embedding service.
    Loads SBERT model once, caches all taxonomy and course embeddings to disk.
    """

    _instance: Optional["EmbeddingService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"[EmbeddingService] Loading SBERT model: {SBERT_MODEL}")
        self.model = SentenceTransformer(SBERT_MODEL)
        self._taxonomy_embeddings: Optional[np.ndarray] = None
        self._taxonomy_skill_names: List[str] = []
        self._course_embeddings: Optional[np.ndarray] = None
        self._course_descriptions: List[str] = []
        self._initialized = True
        logger.info("[EmbeddingService] Model loaded successfully.")

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """
        Encode a list of texts into dense vector embeddings.

        Args:
            texts: List of text strings to encode
            batch_size: Number of texts per batch (memory efficiency)
            normalize: If True, L2-normalize embeddings (unit length for cosine sim)

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings

    def cosine_similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between a query and a matrix of candidates.
        Assumes embeddings are L2-normalized (dot product = cosine similarity).

        Args:
            query_embedding: 1D array of shape (dim,)
            candidate_embeddings: 2D array of shape (n_candidates, dim)

        Returns:
            1D array of cosine similarities of shape (n_candidates,)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        scores = (candidate_embeddings @ query_embedding.T).flatten()
        return scores

    # ─── Taxonomy Embeddings (Precomputed + Cached) ───────────────────────────

    def get_taxonomy_embeddings(self, force_rebuild: bool = False) -> tuple:
        """
        Load or compute embeddings for all skills in the taxonomy.
        Uses disk cache for efficiency.

        Returns:
            (skill_names: List[str], embeddings: np.ndarray)
        """
        if self._taxonomy_embeddings is not None and not force_rebuild:
            return self._taxonomy_skill_names, self._taxonomy_embeddings

        if os.path.exists(EMBEDDING_CACHE_PATH) and not force_rebuild:
            logger.info("[EmbeddingService] Loading taxonomy embeddings from cache.")
            data = np.load(EMBEDDING_CACHE_PATH, allow_pickle=True).item()
            self._taxonomy_skill_names = data["skill_names"]
            self._taxonomy_embeddings = data["embeddings"]
            return self._taxonomy_skill_names, self._taxonomy_embeddings

        logger.info("[EmbeddingService] Building taxonomy embeddings (first-time setup)...")
        with open(TAXONOMY_PATH, "r") as f:
            taxonomy = json.load(f)

        skill_names = []
        skill_texts = []
        for skill in taxonomy["skills"]:
            name = skill["canonical_name"]
            aliases = ", ".join(skill.get("aliases", []))
            category = skill.get("category", "")
            subcategory = skill.get("subcategory", "")
            # Build rich descriptive text for better semantic matching
            text = (
                f"{name}. {aliases}. Category: {category} - {subcategory}. "
                f"Related to: {', '.join(skill.get('related_skills', []))}"
            )
            skill_names.append(name)
            skill_texts.append(text)

        embeddings = self.encode(skill_texts, normalize=True)

        # Save to disk cache
        os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH), exist_ok=True)
        np.save(EMBEDDING_CACHE_PATH, {"skill_names": skill_names, "embeddings": embeddings})

        self._taxonomy_skill_names = skill_names
        self._taxonomy_embeddings = embeddings
        logger.info(f"[EmbeddingService] Taxonomy embeddings built: {len(skill_names)} skills.")
        return skill_names, embeddings

    # ─── Course Embeddings (Precomputed + Cached) ─────────────────────────────

    def get_course_embeddings(self, force_rebuild: bool = False) -> tuple:
        """
        Load or compute embeddings for all courses in the catalog.

        Returns:
            (course_descriptions: List[str], embeddings: np.ndarray)
        """
        if self._course_embeddings is not None and not force_rebuild:
            return self._course_descriptions, self._course_embeddings

        if os.path.exists(COURSE_EMBEDDING_CACHE) and not force_rebuild:
            logger.info("[EmbeddingService] Loading course embeddings from cache.")
            data = np.load(COURSE_EMBEDDING_CACHE, allow_pickle=True).item()
            self._course_descriptions = data["descriptions"]
            self._course_embeddings = data["embeddings"]
            return self._course_descriptions, self._course_embeddings

        logger.info("[EmbeddingService] Building course embeddings...")
        with open(COURSE_CATALOG_PATH, "r") as f:
            catalog = json.load(f)

        descriptions = []
        for course in catalog["courses"]:
            text = (
                f"{course['name']}. {course['description']}. "
                f"Skills: {', '.join(course['skills_covered'])}. "
                f"Level: {course['level']}. Platform: {course['platform']}."
            )
            descriptions.append(text)

        embeddings = self.encode(descriptions, normalize=True)

        os.makedirs(os.path.dirname(COURSE_EMBEDDING_CACHE), exist_ok=True)
        np.save(COURSE_EMBEDDING_CACHE, {"descriptions": descriptions, "embeddings": embeddings})

        self._course_descriptions = descriptions
        self._course_embeddings = embeddings
        logger.info(f"[EmbeddingService] Course embeddings built: {len(descriptions)} courses.")
        return descriptions, embeddings

    def invalidate_cache(self):
        """Force rebuild of all embedding caches."""
        self._taxonomy_embeddings = None
        self._taxonomy_skill_names = []
        self._course_embeddings = None
        self._course_descriptions = []
        for path in [EMBEDDING_CACHE_PATH, COURSE_EMBEDDING_CACHE]:
            if os.path.exists(path):
                os.remove(path)
        logger.info("[EmbeddingService] All embedding caches invalidated.")


# Module-level accessor for easy import
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
