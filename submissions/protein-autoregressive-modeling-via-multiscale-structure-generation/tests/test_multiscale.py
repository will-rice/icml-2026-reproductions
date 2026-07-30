import numpy as np
import pytest
from par_protein.multiscale import MultiscaleDownsampler, ScaleRepresentation


def test_multiscale_downsampling():
    np.random.seed(42)
    coords = np.random.randn(80, 3)
    downsampler = MultiscaleDownsampler(scale_factors=(4, 2, 1))

    scales = downsampler.downsample(coords)
    assert len(scales) == 3
    assert scales[0].num_residues == 20
    assert scales[1].num_residues == 40
    assert scales[2].num_residues == 80


def test_invalid_coords_shape():
    downsampler = MultiscaleDownsampler()
    with pytest.raises(ValueError):
        downsampler.downsample(np.zeros((10, 2)))


def test_coarse_to_fine_map():
    downsampler = MultiscaleDownsampler(scale_factors=(4, 2, 1))
    coords = np.arange(12).reshape(4, 3).astype(float)
    scale0 = ScaleRepresentation(scale_idx=0, num_residues=4, coords=coords, downsample_factor=4)

    mapped = downsampler.coarse_to_fine_map(scale0, fine_length=15)
    assert mapped.shape == (15, 3)
    assert np.array_equal(mapped[0], coords[0])
    assert np.array_equal(mapped[3], coords[0])
    assert np.array_equal(mapped[4], coords[1])
