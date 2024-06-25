from clip.image.vit import ViT, ViTConfig
from clip.languange.gpt import GPT, GPTConfig
from clip.model import CLIP
from clip.model import CLIPWrapper
from clip.loss import CLIPLoss
from clip.data.dataset import CLIPDataset
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader

import torch
import wandb


class Trainer:
    def __init__(self, config):
        self.config = config
        torch.manual_seed(self.config["seed"])
        self.device = self.config["device"]
        self.clip = self.get_model()
        self.loss = CLIPLoss(device=self.device)
        self.model = CLIPWrapper(self.clip, self.loss).to(self.device)
        dataset = CLIPDataset(self.config["dataset"])
        self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        if self.config["optimizer"] == "Adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.config["lr"])  
        else:
            raise ValueError("Only supports Adam.")
        

        if self.config["wandb"]:
            wandb.init(project="clip")

    def get_model(self):
        return CLIP(
            GPT(GPTConfig(device=self.device)),
            ViT(ViTConfig(device=self.device)),
            embd_dim = self.config["embd_dim"],
            temperature = self.config["temperature"],
        ).to(self.device)

    def train(self):
        steps = 1
        for _ in range(self.config["epochs"]):
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                texts, images = batch
                images = images.to(self.device)
                texts = texts.to(self.device)

                loss= self.model(texts, images)
                loss.backward()
                self.optimizer.step()
                if self.config["wandb"] :
                    wandb.log({"loss": loss.item()})
                if steps % self.config["log_freq"] == 0:
                    print(f"Step: {steps}, Loss: {loss.item()}")
                steps += 1

if __name__ == "__main__":
    config = {
        "embd_dim": 512,
        "temperature": 0.2,
        "epochs": 3,
        "batch_size": 32,
        "dataset": f"{Path(__file__).parents[1]}/local_data/raw_data",
        "lr": 3e-5,
        "optimizer": "Adam",
        "seed": 42,
        "device": "cpu",
        "log_freq": 1,
        "wandb": False,


    }
    trainer = Trainer(config)
    trainer.train()

