from fmeh.eval.metrics import classification_scores, extraction_scores


def test_classification_accuracy() -> None:
    metrics = classification_scores("yes", "yes")
    assert metrics["accuracy"] == 1.0


def test_extraction_f1_perfect() -> None:
    target = {"diseases": ["flu"], "chemicals": ["aspirin"]}
    pred = {"diseases": ["flu"], "chemicals": ["aspirin"]}
    metrics = extraction_scores(target, pred)
    assert metrics["f1"] == 1.0
    assert metrics["exact_match"] == 1.0
