import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .config import MODEL_NAME


def chunk_text(text: str, size: int = 120) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.name = "Sentence-BERT (all-MiniLM-L6-v2)"
            # cosine ranges that map to 0 and 1: whole documents / single requirement lines
            self.lo, self.hi = 0.25, 0.65
            self.req_lo, self.req_hi = 0.20, 0.60
        except Exception:  # package or model download unavailable -> still works offline
            self.model = None
            self._hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         n_features=2 ** 15, alternate_sign=False)
            self.name = "Hashing fallback (deep model unavailable)"
            self.lo, self.hi = 0.10, 0.45
            self.req_lo, self.req_hi = 0.10, 0.45

    @property
    def is_deep(self) -> bool:
        return self.model is not None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return self._hv.transform(texts).toarray()

    def doc_vector(self, text: str) -> np.ndarray:
        """Mean of chunk embeddings, L2-normalised: one vector per document."""
        vec = self.encode(chunk_text(text)).mean(axis=0)
        return vec / (np.linalg.norm(vec) or 1.0)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Whole-document cosine similarity rescaled to 0-1."""
        return float(np.clip((float(a @ b) - self.lo) / (self.hi - self.lo), 0, 1))

    def scale_requirement(self, cosine: float) -> float:
        return float(np.clip((cosine - self.req_lo) / (self.req_hi - self.req_lo), 0, 1))
