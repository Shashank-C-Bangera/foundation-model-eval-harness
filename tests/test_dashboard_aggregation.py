import pandas as pd
from sklearn.metrics import f1_score

from fmeh.ui.data import agg_metrics, build_model_leaderboard


def test_macro_f1_uses_slice_aggregation_not_row_mean() -> None:
    y_true = ["yes", "no", "maybe"]
    y_pred = ["yes", "yes", "maybe"]
    df = pd.DataFrame(
        {
            "task": ["classification"] * 3,
            "model_id": ["m1"] * 3,
            "prompt_version": ["v2"] * 3,
            "parse_valid": [True, True, True],
            "invalid_output": [False, False, False],
            "repaired": [False, False, False],
            "exception_occurred": [False, False, False],
            "empty_output": [False, False, False],
            "y_true_norm": y_true,
            "y_pred_norm": y_pred,
            # Deliberately misleading per-row values that should be ignored.
            "macro_f1": [1.0, 1.0, 1.0],
            "accuracy": [1.0, 0.0, 1.0],
        }
    )
    out = agg_metrics(df)
    row = out["by_prompt"].iloc[0]
    expected = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=["yes", "no", "maybe"],
        zero_division=0,
    )
    assert row["macro_f1"] == expected


def test_leaderboard_uses_per_task_n_and_excludes_low_n_task() -> None:
    by_model = pd.DataFrame(
        {
            "model_id": ["m1", "m2"],
            "n_total": [120, 160],
            "parse_valid_rate": [0.9, 0.8],
            "invalid_output_rate": [0.1, 0.2],
            "repair_rate": [0.05, 0.1],
            "exception_rate": [0.0, 0.0],
        }
    )
    by_task_model = pd.DataFrame(
        {
            "task": [
                "classification",
                "summarization",
                "extraction",
                "classification",
                "summarization",
                "extraction",
            ],
            "model_id": ["m1", "m1", "m1", "m2", "m2", "m2"],
            "n": [50, 60, 10, 50, 60, 60],
            "macro_f1": [0.6, pd.NA, pd.NA, 0.5, pd.NA, pd.NA],
            "bertscore_f1": [pd.NA, 0.7, pd.NA, pd.NA, 0.6, pd.NA],
            "extraction_f1": [0.95, pd.NA, 0.95, 0.2, pd.NA, 0.2],
        }
    )
    agg = {"by_model": by_model, "by_task_model": by_task_model, "min_task_n": 50}

    leaderboard = build_model_leaderboard(agg).set_index("model_id")
    assert int(leaderboard.loc["m1", "N_ext"]) == 10
    # m1 extraction is excluded due low N, so score uses cls+sum only.
    assert abs(float(leaderboard.loc["m1", "overall_score"]) - 0.65) < 1e-9
    # m2 includes extraction due sufficient N.
    assert abs(float(leaderboard.loc["m2", "overall_score"]) - ((0.5 + 0.6 + 0.2) / 3)) < 1e-9
