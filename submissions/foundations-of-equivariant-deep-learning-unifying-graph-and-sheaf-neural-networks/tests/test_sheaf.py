import pytest
import torch
import numpy as np
from src.sheaf import SheafLaplacian, SheafGCN, KipfWellingGCN, build_signed_graph
from src.benchmark import run_reproduction_experiments

def test_sheaf_laplacian_identity():
    """Test that identity restriction maps run cleanly and match output dimension."""
    num_nodes, feature_dim = 10, 8
    X = torch.randn(num_nodes, feature_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    sheaf_op = SheafLaplacian(num_nodes, feature_dim, is_identity=True)
    out = sheaf_op(X, edge_index)

    assert out.shape == (num_nodes, feature_dim)
    assert not torch.isnan(out).any()

def test_signed_graph_generation():
    """Test synthetic signed graph generation helper."""
    X, y, edge_index, edge_signs = build_signed_graph(num_nodes=20, num_edges=40, feature_dim=8, num_classes=2, seed=123)
    assert X.shape == (20, 8)
    assert y.shape == (20,)
    assert edge_index.shape[1] == 40
    assert edge_signs.shape[0] == 40
    assert set(edge_signs.numpy().tolist()).issubset({1.0, -1.0})

def test_models_forward_pass():
    """Test forward passes of SheafGCN and KipfWellingGCN."""
    X, y, edge_index, edge_signs = build_signed_graph(num_nodes=15, num_edges=30, feature_dim=8, num_classes=3, seed=42)

    sheaf_model = SheafGCN(in_dim=8, hidden_dim=16, out_dim=3, num_nodes=15)
    out_sheaf = sheaf_model(X, edge_index, edge_signs)
    assert out_sheaf.shape == (15, 3)

    gcn_model = KipfWellingGCN(in_dim=8, hidden_dim=16, out_dim=3, num_nodes=15)
    out_gcn = gcn_model(X, edge_index, edge_signs)
    assert out_gcn.shape == (15, 3)

def test_reproduction_experiments():
    """Test full benchmark pipeline execution."""
    res = run_reproduction_experiments()
    assert res["claim_1_verified"] is True
    assert res["claim_2_verified"] is True
    assert res["claim_3_verified"] is True
    assert res["claim_4_verified"] is True
    assert res["num_trials"] == 5
