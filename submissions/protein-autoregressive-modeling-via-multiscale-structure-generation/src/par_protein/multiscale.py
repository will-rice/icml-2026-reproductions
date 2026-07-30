import sys
sys.dont_write_bytecode = True
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ScaleRepresentation:
    scale_idx: int
    num_residues: int
    coords: np.ndarray
    downsample_factor: int


class MultiscaleDownsampler:
    def __init__(self, scale_factors: tuple[int, ...] = (4, 2, 1)):
        self.scale_factors = scale_factors

    def downsample(self, fine_coords: np.ndarray) -> list[ScaleRepresentation]:
        if fine_coords.ndim != 2 or fine_coords.shape[1] != 3:
            raise ValueError("fine_coords must be 2D array of shape (N, 3)")

        num_fine = fine_coords.shape[0]
        representations = []

        for idx, factor in enumerate(self.scale_factors):
            if factor == 1:
                coarse_coords = fine_coords.copy()
            else:
                num_coarse = int(np.ceil(num_fine / factor))
                coarse_coords = np.zeros((num_coarse, 3), dtype=np.float64)
                for i in range(num_coarse):
                    start = i * factor
                    end = min((i + 1) * factor, num_fine)
                    coarse_coords[i] = np.mean(fine_coords[start:end], axis=0)

            rep = ScaleRepresentation(
                scale_idx=idx,
                num_residues=coarse_coords.shape[0],
                coords=coarse_coords,
                downsample_factor=factor,
            )
            representations.append(rep)

        return representations

    def coarse_to_fine_map(self, coarse_rep: ScaleRepresentation, fine_length: int) -> np.ndarray:
        factor = coarse_rep.downsample_factor
        expanded = np.repeat(coarse_rep.coords, factor, axis=0)
        return expanded[:fine_length]
