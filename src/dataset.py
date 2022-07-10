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

        voxel_coordinates = get_mgrid(*((0, shape) for shape in self.shape))
        self.intensities = self.fdata[
            voxel_coordinates[:, 0],
            voxel_coordinates[:, 1],
            voxel_coordinates[:, 2],
        ].float().unsqueeze(-1)

        # world coordinates in dm
        self.world_coordinates = torch.tensor(
            apply_affine(
                self.affine,
                voxel_coordinates.numpy(),
            )
        ).float() / 100.

        self.batch_size = batch_size
        self.length = (len(self.intensities) - 1) // self.batch_size + 1

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "intensities": self.intensities[idx * self.batch_size : (idx + 1) * self.batch_size],
            "world_coordinates": self.world_coordinates[idx * self.batch_size : (idx + 1) * self.batch_size, :],
        }
