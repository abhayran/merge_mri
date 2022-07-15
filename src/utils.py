import nibabel as nib
import numpy as np
import torch
from typing import List, Tuple
    

def get_mgrid(*coord_ranges: Tuple[int, int]):
    tensors = [
        torch.linspace(
            coord_range[0],
            coord_range[1] - 1,
            steps=coord_range[1] - coord_range[0],
            dtype=torch.long,
        )
        for coord_range in coord_ranges
    ]
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(coord_ranges))
    return mgrid


def get_shifts(affine: np.ndarray) -> np.ndarray:
    mappings = nib.affines.apply_affine(
        affine, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    )
    return np.array(
        [mappings[i + 1, :] - mappings[0, :] for i in range(3)]
    )


def extract_data(*image_paths: str) -> Tuple[List[torch.Tensor], List[np.ndarray]]:    
    fdata_list = [
        torch.tensor(nib.load(image_path).get_fdata())
        for image_path in image_paths
    ]
    affine_list = [
        nib.load(image_path).affine
        for image_path in image_paths
    ]
    voxel_coordinates_list = [
        get_mgrid(*((0, shape) for shape in fdata.shape))
        for fdata in fdata_list
    ]
    world_coordinates_list = [
        torch.tensor(
            nib.affines.apply_affine(
                affine,
                voxel_coordinates.numpy(),
            )
        ).float()
        for affine, voxel_coordinates in zip(affine_list, voxel_coordinates_list)
    ]
    corner_world_coordinates = np.array([
        np.array(torch.max(torch.tensor([
            [torch.min(world_coordinates[:, i]).item() for i in range(3)]
            for world_coordinates in world_coordinates_list
        ]), dim=0)[0]),
        np.array(torch.min(torch.tensor([
            [torch.max(world_coordinates[:, i]).item() for i in range(3)]
            for world_coordinates in world_coordinates_list
        ]), dim=0)[0]),
    ])
    corner_voxel_coordinates_list = [
        nib.affines.apply_affine(
            np.linalg.inv(affine),
            corner_world_coordinates,
        ).astype(int)
        for affine in affine_list
    ]
    fdata_to_return, affine_to_return = list(), list()
    for fdata, affine, corner_voxel_coordinates in zip(fdata_list, affine_list, corner_voxel_coordinates_list):
        close_corner = [min(corner_voxel_coordinates[:, i]) for i in range(3)]
        further_corner = [max(corner_voxel_coordinates[:, i]) for i in range(3)]
        shift = sum(shift * corner_idx for shift, corner_idx in zip(get_shifts(affine), close_corner))
        fdata_to_return.append(
            fdata[
                close_corner[0] : further_corner[0],
                close_corner[1] : further_corner[1],
                close_corner[2] : further_corner[2],
            ]
        )
        affine[:3, -1] += shift
        affine_to_return.append(affine)
    return fdata_to_return, affine_to_return
