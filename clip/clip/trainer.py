from clip.image.vit import ViT, ViTConfig
from clip.languange.gpt import GPT, GPTConfig
from clip.model import CLIP
from clip.model import CLIPWrapper
from clip.loss import CLIPLoss
import json
from clip.data.dataset import CLIPDataset
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader

import torch
import wandb


class Trainer:
    def __init__(self, config):
        torch.manual_seed(config["seed"])
        self.device = config["device"]
        self.clip = self.get_model(config)
        self.loss = CLIPLoss(device=self.device)
        self.model = CLIPWrapper(self.clip, self.loss).to(self.device)
        dataset = CLIPDataset(config["dataset"])
        self.dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        if config["optimizer"] == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=config["lr"])  
        else:
            raise ValueError("Only supports Adam.")
        

        if config["wandb"]:
            wandb.init(project="clip")

    def get_model(self, config):
        return CLIP(
            GPT(GPTConfig(device=self.device)),
            ViT(ViTConfig(device=self.device)),
            embd_dim = config["embd_dim"],
            temperature = config["temperature"],
        ).to(self.device)

    def train(self):
        for _ in range(self.config["steps"]):
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                texts, images = batch
                images = images.to(self.device)
                texts = texts.to(self.device)

                loss= self.model(texts, images)
                loss.backward()
                self.optimizer.step()

                wandb.log({"loss": loss.item()})

if __name__ == "__main__":
    config = {
        "embd_dim": 512,
        "temperature": 0.2,
        "steps": 10000,
        "batch_size": 32,
        "dataset": f"{Path(__file__).parents[1]}/local_data/raw_data",
        "lr": 3e-5,
        "optimizer": "Adam",
        "seed": 42,
        "device": "cpu",
        "wandb": False,


    }
    trainer = Trainer(config)
    trainer.train()

