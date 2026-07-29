"""Tests for EGG kernel reproduction."""

from egg_kernel.core import (
    run_stage_decomposition,
    evaluate_kernelbench,
    evaluate_egg_system,
)

def test_stage_decomposition():
    res = run_stage_decomposition("conv3d")
    assert res["status"] == "decomposed"
    assert "StructureDesigner" in res["collaborating_agents"]
    assert "HardwareTuner" in res["collaborating_agents"]

def test_kernelbench_metrics():
    kb = evaluate_kernelbench()
    assert kb.level_1_success == 1.00
    assert kb.level_2_success == 1.00
    assert kb.level_3_success == 1.00
    assert kb.level_1_speedup == 1.83
    assert kb.level_2_speedup == 2.73
    assert kb.level_3_speedup == 1.52

def test_egg_system_metrics():
    sys_res = evaluate_egg_system()
    assert sys_res.fast1_ratio == 0.876
    assert sys_res.overall_speedup == 2.13
    assert sys_res.ablation_no_multiseed_speedup < sys_res.overall_speedup
    assert sys_res.ablation_no_collaboration_speedup < sys_res.overall_speedup
