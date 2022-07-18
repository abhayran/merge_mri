from tkinter import N
from src.dataset import SirenDataset
from src.models import SirenModel
from src.utils import extract_data, compute_jacobian_loss

import mlflow
import numpy as np
from PIL import Image
import torch

from typing import Dict, Union


class Trainer:
    def __init__(self, config: Dict[str, Union[int, bool]], *image_paths: str) -> None:
        self.config = config
        self.device = torch.device("cuda" if config["use_gpu"] else "cpu")

        fdata_list, affine_list, self.to_normalize_with = extract_data(*image_paths)
        self.datasets = [
            SirenDataset(fdata, affine, batch_size=config["training"]["batch_size"])
            for fdata, affine in zip(fdata_list, affine_list)
        ]
        self.val_data = [
            dataset.get_val_data()
            for dataset in self.datasets
        ]
        self.num_voxels = [
            len(dataset.intensities)
            for dataset in self.datasets
        ]

        self.intensity_model = SirenModel(
            in_features=config["intensity_model"]["input_dim"],
            out_features=1,
            hidden_features=config["intensity_model"]["hidden_dim"], 
            hidden_layers=config["intensity_model"]["num_hidden_layers"], 
            outermost_linear=True, 
            first_omega_0=30, 
            hidden_omega_0=30.
        ).float().to(self.device)

        self.deformable = config["deformable"]
        if self.deformable:
            self.coordinate_mappers = [
                SirenModel(
                    in_features=config["coordinate_mapper"]["input_dim"],
                    out_features=3,
                    hidden_features=config["coordinate_mapper"]["hidden_dim"], 
                    hidden_layers=config["coordinate_mapper"]["num_hidden_layers"], 
                    outermost_linear=True, 
                    first_omega_0=30, 
                    hidden_omega_0=30.
                ).float().to(self.device)
                for _ in range(len(image_paths) - 1)
            ]
            self.regularization_weight = config["coordinate_mapper"]["regularization_weight"]
        self.optimizer = torch.optim.Adam(
            lr=float(config["training"]["lr"]),
            params=list(self.intensity_model.parameters()) 
            + list(self.coordinate_mappers[0].parameters()) 
            + list(self.coordinate_mappers[1].parameters())
            if self.deformable
            else self.intensity_model.parameters()
        )
    
    def log_model(self, epoch: int) -> None:
        mlflow.pytorch.log_model(self.intensity_model, f"intensity_model_{epoch + 1}")
        if self.deformable:
            mlflow.pytorch.log_model(self.coordinate_mappers[0], f"mapper_model_0_epoch_{epoch + 1}")
            mlflow.pytorch.log_model(self.coordinate_mappers[1], f"mapper_model_1_epoch_{epoch + 1}")

    def training_loop_rigid(self) -> float:
        epoch_loss = 0.
        self.optimizer.zero_grad()
        for dataset, num_voxel in zip(self.datasets, self.num_voxels):
            for i in range(len(dataset)):
                item = dataset[i]
                model_output = self.intensity_model(item["world_coordinates"].to(self.device))
                loss = ((model_output * self.to_normalize_with - item["intensities"].to(self.device) * self.to_normalize_with) ** 2).sum() / num_voxel
                epoch_loss += float(loss.detach().item())
                loss.backward()
        self.optimizer.step()
        return epoch_loss
    
    def training_loop_deformable(self) -> float:
        epoch_loss = 0.
        self.optimizer.zero_grad()
        for idx, (dataset, num_voxel) in enumerate(zip(self.datasets, self.num_voxels)):
            for i in range(len(dataset)):
                item = dataset[i]
                if idx == 0:
                    mapped_coordinates = item["world_coordinates"].to(self.device)
                else:
                    world_coordinates = item["world_coordinates"].to(self.device)
                    world_coordinates.requires_grad = True
                    shift = self.coordinate_mappers[idx - 1](world_coordinates)
                    mapped_coordinates = world_coordinates + shift
                    regularization_loss = self.regularization_weight * compute_jacobian_loss(
                        world_coordinates,
                        shift,
                        batch_size=len(world_coordinates),
                    )
                model_output = self.intensity_model(mapped_coordinates)
                loss = ((model_output * self.to_normalize_with - item["intensities"].to(self.device) * self.to_normalize_with) ** 2).sum() / num_voxel
                if idx != 0:
                    loss += regularization_loss
                epoch_loss += float(loss.detach().item())
                loss.backward()
                if idx != 0:
                    mlflow.log_metric("regularization_loss", float(regularization_loss.detach().item()))
        self.optimizer.step()
        return epoch_loss
    
    def training_loop(self) -> float:
        if self.deformable:
            return self.training_loop_deformable()
        return self.training_loop_rigid()

    def validate(self, image_path: str) -> None:
        with torch.no_grad():
            for idx, data in enumerate(self.val_data):
                coordinates = data["world_coordinates"].to(self.device)
                if self.deformable and idx != 0:
                    coordinates += self.coordinate_mappers[idx - 1](coordinates)
                model_output = self.intensity_model(coordinates).view(
                    *self.datasets[idx].shape[:2]
                ).detach().cpu().numpy()
                vis = 255. * np.concatenate((model_output, data["image"]), axis=-1)
                Image.fromarray(vis).convert("L").save(f"{idx}_{image_path}")
                mlflow.log_artifact(f"{idx}_{image_path}")
                os.remove(f"{idx}_{image_path}")


if __name__ == "__main__":
    import os
    import yaml

    image_paths = [
        r"C:\Users\abdul\Desktop\TUM\PMSD\merge_mri\Becken_T2_cor.nii",
        r"C:\Users\abdul\Desktop\TUM\PMSD\merge_mri\Becken_T2_sag.nii",
        r"C:\Users\abdul\Desktop\TUM\PMSD\merge_mri\Becken_T2_tra.nii",
    ]

    with open("src/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    trainer = Trainer(config, *image_paths)
    num_epochs = config["training"]["num_epochs"]
    with mlflow.start_run(run_name=config["mlflow"]["run_name"] + f'_weight_{config["coordinate_mapper"]["regularization_weight"]}'):
        for epoch in range(num_epochs):
            epoch_loss = trainer.training_loop()
            mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
            if (epoch + 1) % config["training"]["image_log_interval"] == 0:
                image_path = f"image_{(len(str(num_epochs)) - len(str(epoch))) * '0'}{epoch}.png"
                trainer.validate(image_path)
            if (epoch + 1) % config["training"]["model_log_interval"] == 0:
                trainer.log_model(epoch + 1)