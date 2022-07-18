from src.utils import get_mgrid, get_shifts

import torch
import nibabel as nib
import numpy as np

from typing import Dict, Union


class SirenDataset(torch.utils.data.Dataset):
    def __init__(self, fdata: torch.Tensor, affine: np.ndarray, batch_size: int) -> None:
        self.fdata = fdata
        self.shape = tuple(self.fdata.shape)
        assert len(self.shape) == 3
        
        self.affine = affine
        self.shifts = get_shifts(affine)
        
        self.voxel_coordinates = get_mgrid(*((0, shape) for shape in self.shape))
        self.intensities = self.fdata[
            self.voxel_coordinates[:, 0],
            self.voxel_coordinates[:, 1],
            self.voxel_coordinates[:, 2],
        ].float().unsqueeze(-1)

        # world coordinates in dm
        self.world_coordinates = torch.tensor(
            nib.affines.apply_affine(
                self.affine,
                self.voxel_coordinates.numpy(),
            )
        ).float() / 100.

        self.batch_size = batch_size
        self.length = (len(self.intensities) - 1) // self.batch_size + 1

    def get_val_data(self, to_div=None) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        if to_div is None:
            to_div = 2
        voxel_coordinates = get_mgrid(
            (0, self.shape[0]),
            (0, self.shape[1]),
            (int(self.shape[2] // to_div), int(self.shape[2] // to_div) + 1)
        )
        world_coordinates = torch.tensor(
            nib.affines.apply_affine(
                self.affine,
                voxel_coordinates.numpy(),
            )
        ).float() / 100.
        return {
            "world_coordinates": world_coordinates,
            "image": self.fdata[:, :, int(self.shape[2] // to_div)]
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "intensities": self.intensities[idx * self.batch_size : (idx + 1) * self.batch_size],
            "world_coordinates": self.world_coordinates[idx * self.batch_size : (idx + 1) * self.batch_size, :],
        }
