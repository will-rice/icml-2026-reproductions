import numpy as np
from par_protein.multiscale import ScaleRepresentation
from par_protein.model import AutoregressiveTransformer, FlowBackboneDecoder, PARModel


def test_autoregressive_transformer():
    transformer = AutoregressiveTransformer(embed_dim=32)
    coords = np.random.randn(10, 3)
    scale_rep = ScaleRepresentation(scale_idx=0, num_residues=10, coords=coords, downsample_factor=2)

    embeds = transformer.compute_conditional_embeddings(scale_rep, target_length=20)
    assert embeds.shape == (20, 32)
    assert not np.isnan(embeds).any()


def test_flow_backbone_decoder():
    decoder = FlowBackboneDecoder(num_steps=5)
    cond_embeds = np.random.randn(15, 32)

    decoded = decoder.decode(cond_embeds)
    assert decoded.shape == (15, 3)
    assert not np.isnan(decoded).any()


def test_par_model_end_to_end():
    model = PARModel(scale_factors=(4, 2, 1), embed_dim=32)
    init_coords = np.random.randn(40, 3)

    outputs = model.generate_backbone(init_coords)
    assert "scale_0" in outputs
    assert "scale_1" in outputs
    assert "scale_2" in outputs
    assert outputs["scale_0"].shape == (10, 3)
    assert outputs["scale_1"].shape == (20, 3)
    assert outputs["scale_2"].shape == (40, 3)
