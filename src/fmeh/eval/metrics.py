from __future__ import annotations

import math
from typing import Any

import evaluate
from sklearn.metrics import f1_score

from fmeh.eval.validators import normalize_label

_ROUGE = None
_BERTSCORE = None


def _get_rouge():
    global _ROUGE
    if _ROUGE is None:
        _ROUGE = evaluate.load("rouge")
    return _ROUGE


def _get_bertscore():
    global _BERTSCORE
    if _BERTSCORE is None:
        _BERTSCORE = evaluate.load("bertscore")
    return _BERTSCORE


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def extraction_scores(target: dict[str, Any], pred: dict[str, Any]) -> dict[str, float]:
    target_d = {x.strip().lower() for x in target.get("diseases", []) if isinstance(x, str)}
    target_c = {x.strip().lower() for x in target.get("chemicals", []) if isinstance(x, str)}
    pred_d = {x.strip().lower() for x in pred.get("diseases", []) if isinstance(x, str)}
    pred_c = {x.strip().lower() for x in pred.get("chemicals", []) if isinstance(x, str)}

    target_all = target_d | target_c
    pred_all = pred_d | pred_c
    tp = len(target_all & pred_all)
    precision = _safe_div(tp, len(pred_all))
    recall = _safe_div(tp, len(target_all))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    exact = float(target_d == pred_d and target_c == pred_c)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact,
    }


def summarize_scores(target: str, pred: str, source: str) -> dict[str, float]:
    rouge = _get_rouge()
    rouge_result = rouge.compute(predictions=[pred], references=[target])

    bert_f1 = math.nan
    try:
        bert = _get_bertscore()
        bert_result = bert.compute(predictions=[pred], references=[target], lang="en")
        bert_f1 = float(bert_result["f1"][0])
    except Exception:
        bert_f1 = math.nan

    pred_words = max(len(pred.split()), 1)
    source_words = max(len(source.split()), 1)

    return {
        "rougeL": float(rouge_result.get("rougeL", 0.0)),
        "bertscore_f1": bert_f1,
        "pred_length": float(pred_words),
        "compression_ratio": float(pred_words / source_words),
    }


def classification_scores(target_label: str, pred_label: str) -> dict[str, float]:
    t = normalize_label(target_label)
    p = normalize_label(pred_label)
    acc = float(t == p)
    macro = f1_score([t], [p], average="macro", labels=["yes", "no", "maybe"], zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1_single": float(macro),
    }


def unsupported_claim_proxy(summary: str, context: str) -> float:
    summary_terms = {t.lower() for t in summary.split() if len(t) > 4}
    context_terms = {t.lower() for t in context.split()}
    if not summary_terms:
        return 0.0
    unsupported = len(summary_terms - context_terms)
    return float(unsupported / len(summary_terms))
