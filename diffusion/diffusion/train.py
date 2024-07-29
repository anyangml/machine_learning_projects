import torch
from diffusion import Diffusion
from diffusion import UNetWithTime
from data.dataset import SwissRollDataset, DataLoader
from pathlib import Path
from torch.optim import Adam

import torch
class Trainer:
    def __init__(self) -> None:
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Diffusion(UNetWithTime()).to(self.device)
        dataset = SwissRollDataset(list(Path(f"{Path(__file__).parents[0]}/data/dummy_data").rglob("*.png")))
        self.dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        self.optimizer = Adam(self.model.parameters(), lr=2e-4)  

    def train(self):
        steps = 1
        for _ in range(100):
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                images = batch
                images = images.to(self.device)

                loss= self.model(images)
                print(loss)
                loss.backward()
                self.optimizer.step()
                # if self.config["wandb"] :
                #     wandb.log({"loss": loss.item()})
                if steps % 20 == 0:
                    print(f"Step: {steps}, Loss: {loss.item()}")
                steps += 1

if __name__ == "__main__":
    trainer = Trainer()
    with torch.autograd.set_detect_anomaly(True):
        trainer.train()