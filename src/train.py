from src.dataset import SirenDataset
from src.models import SirenModel

import nibabel as nib
import numpy as np
import torch

from typing import Dict, Union


class Trainer:
    def __init__(self, image_path: str, config: Dict[str, Union[int, bool]]) -> None:
        self.config = config
        self.device = torch.device("cuda" if config["use_gpu"] else "cpu")
        image = nib.load(image_path)
        self.fdata = image.get_fdata()
        self.affine = image.affine
        self.dataloader = torch.utils.data.DataLoader(
            SirenDataset(self.fdata, self.affine, batch_size=config["training"]["batch_size"]),
            shuffle=False,
            batch_size=1
        )
        self.num_voxels = len(self.dataloader.dataset.intensities)
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
    
    def training_loop(self, return_pred: bool = False) -> Dict[str, Union[float, np.ndarray]]:
        epoch_loss = 0.
        self.optimizer.zero_grad()
        if return_pred:
            pred = list()
        for item in self.dataloader:
            model_output = self.model(item["world_coordinates"].to(self.device))
            loss = ((model_output - item["intensities"].to(self.device)) ** 2).sum() / self.num_voxels
            epoch_loss += float(loss.detach().item())
            if return_pred:
                pred.append(model_output.detach().squeeze().cpu())
            loss.backward()
        self.optimizer.step()
        if return_pred:
            return {
                "epoch_loss": epoch_loss,
                "pred": torch.cat(pred),
            }
        else:
            return {
                "epoch_loss": epoch_loss,
            }


if __name__ == "__main__":
    import mlflow
    import os
    from PIL import Image
    import yaml

    image_path = r"C:\Users\abdul\Desktop\TUM\PMSD\merge_mri\T2W_2.nii"

    with open("src/config.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    trainer = Trainer(image_path, config)
    with mlflow.start_run(run_name=config["mlflow"]["run_name"]):
        for epoch in range(config["training"]["num_epochs"]):
            log_image = epoch % config["training"]["image_log_interval"] == 0
            epoch_summary = trainer.training_loop(
                return_pred=log_image
            )
            mlflow.log_metric("epoch_loss", epoch_summary["epoch_loss"], step=epoch)
            if log_image:
                image_path = os.path.join("images", f"image_{epoch}.png")
                Image.fromarray(
                    epoch_summary["pred"].view(*trainer.dataloader.dataset.shape)[:, :, 0].detach().numpy()
                ).convert("L").save(image_path)
                mlflow.log_artifact(image_path)
                os.remove(image_path)
