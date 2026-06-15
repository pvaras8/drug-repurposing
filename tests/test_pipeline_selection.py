from repurposing_pipeline.pipeline import _select_boltz_candidates


def test_select_boltz_candidates_caps_negative_scores_to_best_70() -> None:
    scores = [(f"mol_{index:03d}", float(-index)) for index in range(1, 401)]

    selected, threshold = _select_boltz_candidates(scores, max_molecules=70)

    assert len(selected) == 70
    assert selected == {f"mol_{index:03d}" for index in range(331, 401)}
    assert threshold == -300.25


def test_select_boltz_candidates_keeps_all_when_quartile_is_smaller_than_limit() -> None:
    scores = [(f"mol_{index}", float(-index)) for index in range(1, 9)]

    selected, threshold = _select_boltz_candidates(scores, max_molecules=70)

    assert selected == {"mol_7", "mol_8"}
    assert threshold == -6.25
