from fmeh.graph.nodes import _parse_output


def test_parse_valid_classification_json() -> None:
    raw = '{"label":"yes","rationale":"supported"}'
    parsed, err, empty = _parse_output("classification", raw)
    assert err == ""
    assert empty is False
    assert parsed is not None
    assert parsed["label"] == "yes"


def test_parse_repair_single_quotes() -> None:
    raw = "{'diseases':['flu'],'chemicals':['aspirin'],}"
    parsed, err, empty = _parse_output("extraction", raw)
    assert err == ""
    assert empty is False
    assert parsed is not None
    assert parsed["diseases"] == ["flu"]
