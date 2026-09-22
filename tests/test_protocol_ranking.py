"""Protocol-level checks without invoking docking or Boltz executables."""

from repurposing_pipeline.ranking import apply_ranking, score_boltz


def test_single_target_threshold_and_score():
    rows = [
        {"molecule_id": "pass", "boltz_status": "completed", "boltz_affinity_pred_value": -2, "boltz_affinity_probability_binary": 0.45},
        {"molecule_id": "low", "boltz_status": "completed", "boltz_affinity_pred_value": -10, "boltz_affinity_probability_binary": 0.44},
        {"molecule_id": "nonnegative", "boltz_status": "completed", "boltz_affinity_pred_value": 0, "boltz_affinity_probability_binary": 0.9},
    ]
    ranked = apply_ranking(rows)
    assert ranked[0]["pIC50"] == 8
    assert ranked[0]["final_score"] == 3.6
    assert ranked[0]["final_rank"] == 1
    assert all(row["final_rank"] is None for row in ranked[1:])


def test_two_target_geometric_mean_and_all_target_gate():
    row = {"molecule_id": "both", "a_boltz_status": "completed", "a_boltz_affinity_pred_value": -2, "a_boltz_affinity_probability_binary": 0.5, "b_boltz_status": "completed", "b_boltz_affinity_pred_value": -3, "b_boltz_affinity_probability_binary": 0.6}
    ranked = apply_ranking([row], ["a", "b"])
    assert ranked[0]["a_pIC50"] == 8
    assert ranked[0]["b_pIC50"] == 9
    assert round(ranked[0]["final_score"], 6) == round((4 * 5.4) ** 0.5, 6)
    row["b_boltz_affinity_probability_binary"] = 0.4
    assert score_boltz(row, prefix="b_") is None
    assert apply_ranking([row], ["a", "b"])[0]["final_rank"] is None
