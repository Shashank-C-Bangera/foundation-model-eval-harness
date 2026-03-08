from sklearn.metrics import f1_score

from fmeh.eval.metrics import classification_scores, classification_slice_scores, extraction_scores
from fmeh.eval.validators import normalize_label


def test_classification_accuracy() -> None:
    metrics = classification_scores("yes", "yes")
    assert metrics["accuracy"] == 1.0


def test_extraction_f1_perfect() -> None:
    target = {"diseases": ["flu"], "chemicals": ["aspirin"]}
    pred = {"diseases": ["flu"], "chemicals": ["aspirin"]}
    metrics = extraction_scores(target, pred)
    assert metrics["f1"] == 1.0
    assert metrics["exact_match"] == 1.0


def test_label_normalization_char_list_yes() -> None:
    assert normalize_label(["y", "e", "s"]) == "yes"
    assert normalize_label("Y e S") == "yes"


def test_classification_slice_macro_f1() -> None:
    y_true = ["yes", "no", "maybe"]
    y_pred = ["yes", "yes", "maybe"]
    metrics = classification_slice_scores(y_true, y_pred)
    expected = f1_score(
        y_true, y_pred, average="macro", labels=["yes", "no", "maybe"], zero_division=0
    )
    assert metrics["macro_f1"] == expected
