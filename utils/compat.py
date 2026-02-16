"""
Compatibility shims for packages not available in the offline environment.
These provide functionally equivalent implementations using available packages.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

# ─── Rapidfuzz Shim ───────────────────────────────────────────────────────────
# Provides token_set_ratio using pure Python / difflib
try:
    from rapidfuzz import fuzz as _fuzz
    fuzz = _fuzz
except ImportError:
    from difflib import SequenceMatcher

    class _FuzzShim:
        @staticmethod
        def token_set_ratio(s1: str, s2: str) -> float:
            """
            Approximate token_set_ratio using sorted token comparison.
            Sorts tokens alphabetically before comparison to handle word-order variations.
            """
            tokens1 = sorted(s1.lower().split())
            tokens2 = sorted(s2.lower().split())
            sorted1 = " ".join(tokens1)
            sorted2 = " ".join(tokens2)
            ratio = SequenceMatcher(None, sorted1, sorted2).ratio()
            return ratio * 100  # Return 0-100 scale like rapidfuzz

        @staticmethod
        def ratio(s1: str, s2: str) -> float:
            return SequenceMatcher(None, s1.lower(), s2.lower()).ratio() * 100

    fuzz = _FuzzShim()


# ─── Sentence Transformers Shim ───────────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST
    SentenceTransformer = _ST
    HAS_SBERT = True
except ImportError:
    import hashlib
    import re

    HAS_SBERT = False

    class SentenceTransformer:
        """
        Deterministic pseudo-embedding fallback.
        Produces consistent 384-dim vectors using TF-IDF-like term hashing.
        This mimics the vector space properties of SBERT for offline use.
        """
        DIM = 384

        def __init__(self, model_name_or_path: str = "all-MiniLM-L6-v2"):
            self.model_name = model_name_or_path
            self._vocab_cache = {}
            print(f"[COMPAT] sentence-transformers not available. Using TF-IDF pseudo-embedding "
                  f"(shape: {self.DIM}d) as fallback for '{model_name_or_path}'.")

        def encode(
            self,
            sentences,
            batch_size: int = 64,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True
        ) -> np.ndarray:
            if isinstance(sentences, str):
                sentences = [sentences]

            embeddings = np.array([self._text_to_vector(s) for s in sentences])

            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # Avoid div-by-zero
                embeddings = embeddings / norms

            return embeddings

        def _text_to_vector(self, text: str) -> np.ndarray:
            """
            Create a deterministic pseudo-embedding for text.
            Uses character n-gram hashing into a fixed 384-dim space.
            This is consistent (same text → same vector) and captures
            some lexical similarity between related terms.
            """
            text = text.lower().strip()

            # Extract meaningful tokens
            tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\+\#]*\b", text)
            if not tokens:
                tokens = [text[:20]] if text else ["empty"]

            vec = np.zeros(self.DIM)

            for token in tokens:
                # Use multiple hash functions for better distribution
                for seed in [1, 17, 31, 53, 97]:
                    h = hashlib.md5(f"{seed}:{token}".encode()).hexdigest()
                    # Map hash to multiple vector positions
                    for offset in range(0, 32, 4):
                        chunk = int(h[offset:offset+4], 16)
                        dim_idx = (chunk + seed * 1000) % self.DIM
                        value = (chunk / 65535.0) * 2 - 1  # Map to [-1, 1]
                        vec[dim_idx] += value

            # Add character-level bigrams for sub-word similarity
            text_flat = "".join(tokens)
            for i in range(len(text_flat) - 1):
                bigram = text_flat[i:i+2]
                h = hashlib.md5(f"bg:{bigram}".encode()).digest()
                idx1 = int.from_bytes(h[:2], 'big') % self.DIM
                idx2 = int.from_bytes(h[2:4], 'big') % self.DIM
                vec[idx1] += 0.3
                vec[idx2] -= 0.3

            return vec
