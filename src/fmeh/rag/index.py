from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from fmeh.config import HarnessConfig


def build_index(cfg: HarnessConfig, source_df: pd.DataFrame) -> None:
    passages = source_df[source_df["task"].isin(["classification", "summarization"])][
        ["example_id", "input"]
    ].drop_duplicates()

    passages = passages.rename(columns={"input": "text"})
    if passages.empty:
        return

    model = SentenceTransformer(cfg.rag.embedding_model)
    vectors = model.encode(
        passages["text"].tolist(), convert_to_numpy=True, show_progress_bar=False
    )
    vectors = vectors.astype(np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    faiss.normalize_L2(vectors)
    index.add(vectors)

    index_path = Path(cfg.rag.index_path)
    passages_path = Path(cfg.rag.passages_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    passages_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    passages.to_parquet(passages_path, index=False)
