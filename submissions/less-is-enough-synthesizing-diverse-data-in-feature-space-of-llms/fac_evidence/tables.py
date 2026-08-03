import re


METRIC_COLUMNS = [
    "toxicity_auprc",
    "reward_avg_acc",
    "steering_sycophancy_scr",
    "steering_survival_scr",
    "instruction_lc",
    "instruction_wr",
    "instruction_sd",
]


def _plain_latex(text: str) -> str:
    return (
        text.replace("\\textbf{", "")
        .replace("\\%", "%")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
    )


def _first_float(cell: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", _plain_latex(cell))
    if not match:
        raise ValueError(f"no numeric value found in table cell: {cell!r}")
    return float(match.group(0))


def parse_feature_ratio_table(tex: str) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    for line in tex.splitlines():
        match = re.match(r"\s*(30|60|100)\\?%\s*&(.+?)\\\\", line)
        if not match:
            continue
        ratio = int(match.group(1))
        cells = [cell.strip() for cell in match.group(2).split("&")]
        if len(cells) != len(METRIC_COLUMNS):
            raise ValueError(f"expected {len(METRIC_COLUMNS)} metric cells for {ratio}%, got {len(cells)}")
        rows[ratio] = {metric: _first_float(cell) for metric, cell in zip(METRIC_COLUMNS, cells, strict=True)}
    required = {30, 60, 100}
    if set(rows) != required:
        raise ValueError(f"missing feature-ratio rows: {sorted(required - set(rows))}")
    return rows


def main_result_context() -> dict[str, dict[str, float]]:
    return {
        "baseline": {
            "toxicity_auprc": 38.97,
            "reward_avg_acc": 62.90,
            "steering_sycophancy_scr": 16.67,
            "steering_survival_scr": -2.00,
            "instruction_lc": 1.80,
            "instruction_wr": 1.80,
        },
        "ours": {
            "toxicity_auprc": 62.60,
            "reward_avg_acc": 76.22,
            "steering_sycophancy_scr": 40.67,
            "steering_survival_scr": 40.00,
            "instruction_lc": 20.27,
            "instruction_wr": 21.26,
        },
    }
