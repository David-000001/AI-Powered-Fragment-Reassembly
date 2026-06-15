"""
embeddings.py
-------------
Utility for computing sentence embeddings using sentence-transformers.
Falls back to TF-IDF if the model cannot be loaded (offline/low-resource env).
"""

from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class EmbeddingModel:
    """
    Wraps sentence-transformers with a TF-IDF fallback.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for sentence-transformers.
    use_fallback : bool
        Force TF-IDF fallback even if sentence-transformers is available.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_fallback: bool = False):
        self.model_name = model_name
        self._model = None
        self._tfidf = None
        self._use_fallback = use_fallback
        self._load_model()

    def _load_model(self) -> None:
        if self._use_fallback:
            print("[EmbeddingModel] Using TF-IDF fallback.")
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            print(f"[EmbeddingModel] Loaded '{self.model_name}' successfully.")
        except Exception as e:
            print(f"[EmbeddingModel] Could not load sentence-transformers ({e}). Falling back to TF-IDF.")
            self._use_fallback = True

    def encode(self, sentences: list[str]) -> np.ndarray:
        """
        Encode a list of sentences into a 2-D numpy array of shape (N, D).
        """
        if not self._use_fallback and self._model is not None:
            embeddings = self._model.encode(sentences, show_progress_bar=False)
            return np.array(embeddings, dtype=np.float32)

        # TF-IDF fallback
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer()
            matrix = self._tfidf.fit_transform(sentences).toarray()
        else:
            matrix = self._tfidf.transform(sentences).toarray()
        return normalize(matrix, norm="l2").astype(np.float32)
