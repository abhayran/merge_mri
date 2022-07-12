from src.dataset import SirenDataset
from src.models import SirenModel

import mlflow
import numpy as np
from PIL import Image
import torch

from typing import Dict, Union


class Trainer:
    def __init__(self, config: Dict[str, Union[int, bool]], *image_paths: str) -> None:
        self.config = config
        self.device = torch.device("cuda" if config["use_gpu"] else "cpu")

        self.dataloaders = [
            torch.utils.data.DataLoader(
                SirenDataset(image_path, batch_size=config["training"]["batch_size"]),
                shuffle=False,
                batch_size=1,
            )
            for image_path in image_paths
        ]
        self.val_data = [
            dataloader.dataset.get_val_data()
            for dataloader in self.dataloaders
        ]
        self.num_voxels = [
            len(dataloader.dataset.intensities)
            for dataloader in self.dataloaders
        ]

        self.model = SirenModel(
            in_features=config["model"]["input_dim"],
            out_features=1,
            hidden_features=config["model"]["hidden_dim"], 
            hidden_layers=config["model"]["num_hidden_layers"], 
            outermost_linear=True, 
            first_omega_0=30, 
            hidden_omega_0=30.
        ).float().to(self.device)
        self.optimizer = torch.optim.Adam(lr=float(config["training"]["lr"]), params=self.model.parameters())
    
    def training_loop(self) -> float:
        epoch_loss = 0.
        self.optimizer.zero_grad()
        for dataloader, num_voxel in zip(self.dataloaders, self.num_voxels):
            for item in dataloader:
                model_output = self.model(item["world_coordinates"].to(self.device))
                loss = ((model_output - item["intensities"].to(self.device)) ** 2).sum() / num_voxel
                epoch_loss += float(loss.detach().item())
                loss.backward()
        self.optimizer.step()
        return epoch_loss
    
    def validate(self, image_path: str) -> None:
        with torch.no_grad():
            for idx, data in enumerate(self.val_data): 
                model_output = self.model(
                    data["world_coordinates"].to(self.device)
                ).view(*self.dataloaders[idx].dataset.shape[:2]).detach().cpu().numpy()
                vis = np.concatenate((model_output, data["image"]), axis=-1)
                Image.fromarray(vis).convert("L").save(f"{idx}_{image_path}")
                mlflow.log_artifact(f"{idx}_{image_path}")
                os.remove(f"{idx}_{image_path}")


if __name__ == "__main__":
    import os
    import yaml

    image_path_1 = r"C:\Users\abdul\Desktop\TUM\PMSD\merge_mri\T2W_1.nii"
    image_path_2 = r"C:\Users\abdul\Desktop\TUM\PMSD\merge_mri\T2W_2.nii"

    with open("src/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    trainer = Trainer(config, image_path_1, image_path_2)
    num_epochs = config["training"]["num_epochs"]
    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        for epoch in range(num_epochs):
            log_image = epoch % config["training"]["image_log_interval"] == 0
            epoch_loss = trainer.training_loop()
            mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
            if log_image:
                image_path = f"image_{(len(str(num_epochs)) - len(str(epoch))) * '0'}{epoch}.png"
                trainer.validate(image_path)
        mlflow.pytorch.log_model(trainer.model, "model")
