from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class FaissRetriever:
    def __init__(self, index_path: str, passages_path: str, embedding_model: str) -> None:
        self.index_path = Path(index_path)
        self.passages_path = Path(passages_path)
        if not self.index_path.exists() or not self.passages_path.exists():
            raise FileNotFoundError("RAG index or passages file missing")

        self.index = faiss.read_index(str(self.index_path))
        self.passages = pd.read_parquet(self.passages_path)
        self.encoder = SentenceTransformer(embedding_model)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        vec = self.encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, top_k)

        rows = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.passages):
                continue
            row = self.passages.iloc[int(idx)]
            rows.append(
                {
                    "example_id": str(row.get("example_id", "")),
                    "text": str(row.get("text", "")),
                    "score": float(score),
                }
            )
        return rows
