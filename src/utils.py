import torch
from typing import Tuple
    

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
