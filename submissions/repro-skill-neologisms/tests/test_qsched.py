import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
from qsched.scheduler import QSchedScheduler
from qsched.jaq_loss import JAQLoss
from qsched.eval import run_evaluation

def test_qsched_scheduler_initialization():
    scheduler = QSchedScheduler(num_timesteps=1000, bit_width_w=4, bit_width_a=8)
    assert scheduler.num_timesteps == 1000
    assert scheduler.bit_width_w == 4
    assert scheduler.bit_width_a == 8

def test_qsched_scheduler_quantization_noise():
    scheduler = QSchedScheduler()
    x = torch.randn(10, 10)
    noise = scheduler.get_quantization_noise_bound(x, bits=4)
    assert isinstance(noise, float)
    assert noise >= 0.0

def test_qsched_optimize_schedule():
    scheduler = QSchedScheduler()
    steps_4 = scheduler.optimize_few_step_schedule(num_inference_steps=4)
    assert len(steps_4) == 4
    assert steps_4[-1] < 1000
    assert (np.diff(steps_4) >= 0).all()

def test_jaq_loss_computation():
    jaq = JAQLoss(lambda_align=0.5, lambda_quality=0.5)
    text_emb = torch.randn(8, 512)
    img_feat = torch.randn(8, 512)
    jaq_results = jaq(text_emb, img_feat)
    assert "total_loss" in jaq_results
    assert "alignment_score" in jaq_results
    assert "quality_score" in jaq_results

def test_run_evaluation():
    res = run_evaluation()
    assert res["status"] == "success"
    assert res["paper_id"] == "4yzY0GFIJj"
    assert len(res["claims_verified"]) >= 2
