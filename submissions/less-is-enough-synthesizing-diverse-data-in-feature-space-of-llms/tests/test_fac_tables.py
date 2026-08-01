from pathlib import Path

from fac_evidence.tables import parse_feature_ratio_table


def test_feature_ratio_table_monotonicity_is_computed_not_assumed():
    tex = (Path(__file__).parent / "fixtures" / "fac_feature_ratio.tex").read_text()

    rows = parse_feature_ratio_table(tex)

    assert rows[30]["toxicity_auprc"] == 45.60
    assert rows[100]["reward_avg_acc"] == 74.76
    for metric in [
        "toxicity_auprc",
        "reward_avg_acc",
        "steering_sycophancy_scr",
        "steering_survival_scr",
        "instruction_wr",
    ]:
        assert rows[100][metric] >= rows[60][metric] >= rows[30][metric]
