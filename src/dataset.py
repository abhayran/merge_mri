from src.utils import get_mgrid

import torch
import nibabel as nib
import numpy as np

from typing import Dict, Union


class SirenDataset(torch.utils.data.Dataset):
    def __init__(self, image_path: str, batch_size: int) -> None:
        image = nib.load(image_path)
        fdata = image.get_fdata()
        affine = image.affine
        
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
            nib.affines.apply_affine(
                self.affine,
                self.voxel_coordinates.numpy(),
            )
        ).float() / 100.

        self.batch_size = batch_size
        self.length = (len(self.intensities) - 1) // self.batch_size + 1

        mappings = nib.affines.apply_affine(
            self.affine, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        )
        self.shifts = np.array(
            [mappings[i + 1, :] - mappings[0, :] for i in range(3)]
        )

    def get_val_data(self) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        voxel_coordinates = get_mgrid(
            (0, self.shape[0]),
            (0, self.shape[1]),
            (self.shape[2] // 2, self.shape[2] // 2 + 1)
        )
        world_coordinates = torch.tensor(
            nib.affines.apply_affine(
                self.affine,
                voxel_coordinates.numpy(),
            )
        ).float() / 100.
        return {
            "world_coordinates": world_coordinates,
            "image": self.fdata[:, :, self.shape[2] // 2]
        }

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "intensities": self.intensities[idx * self.batch_size : (idx + 1) * self.batch_size],
            "world_coordinates": self.world_coordinates[idx * self.batch_size : (idx + 1) * self.batch_size, :],
        }
