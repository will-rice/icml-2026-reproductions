"""Evaluation correlation utilities for DeMix, matching proxy_eval.py."""

from typing import Dict, Tuple, Any
import math
from scipy import stats
import numpy as np

def get_top_spearman(pred_scores: list, gt_scores: list, top_k: float = 0.25) -> float:
    """Calculate Spearman rank correlation on the top-k% ground truth performers."""
    if pred_scores is None or gt_scores is None:
        return 0.0
    if len(pred_scores) != len(gt_scores) or len(gt_scores) < 2:
        return 0.0

    n = len(gt_scores)
    k = max(2, int(n * top_k))
    k = min(k, n)

    gt_arr = np.asarray(gt_scores, dtype=float)
    pred_arr = np.asarray(pred_scores, dtype=float)

    top_idx = np.argsort(gt_arr)[-k:]

    rho, _ = stats.spearmanr(pred_arr[top_idx], gt_arr[top_idx])
    if math.isnan(float(rho)):
        return 0.0
    return float(rho)

def eval_correlations(
    pred_data_dict: Dict[str, Dict[str, float]],
    gt_data_dict: Dict[str, Dict[str, float]]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Evaluate Spearman correlations between proxy and ground truth benchmark scores."""
    mixture_ids = list(pred_data_dict.keys())
    if not mixture_ids or pred_data_dict[mixture_ids[0]] is None:
        return {}, {}, {}

    benchmarks = list(pred_data_dict[mixture_ids[0]].keys())

    rho_domain_dict = {}
    top_25_rho_domain_dict = {}
    maintain_domain_dict = {}

    for benchmark in benchmarks:
        gt_scores = []
        pred_scores = []

        for mixture_id in mixture_ids:
            if pred_data_dict.get(mixture_id) and gt_data_dict.get(mixture_id):
                gt_scores.append(gt_data_dict[mixture_id].get(benchmark, 0.0))
                pred_scores.append(pred_data_dict[mixture_id].get(benchmark, 0.0))

        if len(gt_scores) < 2:
            continue

        rho, _ = stats.spearmanr(pred_scores, gt_scores)
        top_25_rho = get_top_spearman(pred_scores, gt_scores)

        if math.isnan(float(rho)):
            rho = 0.0

        if '_avg' in benchmark or benchmark == 'avg':
            rho_domain_dict[benchmark] = float(rho)
            top_25_rho_domain_dict[benchmark] = float(top_25_rho)
            sum_gt = sum(gt_scores)
            maintain_domain_dict[benchmark] = float(sum(pred_scores) / sum_gt) if sum_gt != 0 else 0.0

    if rho_domain_dict:
        rho_domain_dict['avg'] = float(sum(rho_domain_dict.values()) / len(rho_domain_dict))
        top_25_rho_domain_dict['avg'] = float(sum(top_25_rho_domain_dict.values()) / len(top_25_rho_domain_dict))
        maintain_domain_dict['avg'] = float(sum(maintain_domain_dict.values()) / len(maintain_domain_dict))

    return rho_domain_dict, top_25_rho_domain_dict, maintain_domain_dict
