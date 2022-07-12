from src.utils import get_mgrid

import torch
from nibabel.affines import apply_affine
import numpy as np
from typing import Dict


class SirenDataset(torch.utils.data.Dataset):
    def __init__(self, fdata: np.ndarray, affine: np.ndarray, batch_size: int) -> None:
        self.fdata = torch.tensor(fdata)
        self.affine = affine
        self.shape = tuple(self.fdata.shape)
        assert len(self.shape) == 3

        self.voxel_coordinates = get_mgrid(*((0, shape) for shape in self.shape))
        self.intensities = self.fdata[
            self.voxel_coordinates[:, 0],
            self.voxel_coordinates[:, 1],
            self.voxel_coordinates[:, 2],
        ].float().unsqueeze(-1)

        # world coordinates in dm
        self.world_coordinates = torch.tensor(
            apply_affine(
                self.affine,
                self.voxel_coordinates.numpy(),
            )
        ).float() / 100.

        self.batch_size = batch_size
        self.length = (len(self.intensities) - 1) // self.batch_size + 1

        mappings = apply_affine(
            self.affine, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        )
        self.shifts = np.array(
            [mappings[i + 1, :] - mappings[0, :] for i in range(3)]
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "intensities": self.intensities[idx * self.batch_size : (idx + 1) * self.batch_size],
            "world_coordinates": self.world_coordinates[idx * self.batch_size : (idx + 1) * self.batch_size, :],
        }
