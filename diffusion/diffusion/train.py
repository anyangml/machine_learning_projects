import torch
from diffusion_wrap import Diffusion
from diffusion_wrap import UNetWithTime
from data.dataset import SwissRollDataset, DataLoader
from pathlib import Path
from torch.optim import Adam


import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self) -> None:
        torch.manual_seed(412)
        self.model = Diffusion(UNetWithTime()).to(DEVICE)

        dataset = SwissRollDataset(list(Path(f"{Path(__file__).parents[0]}/data/dummy_data").rglob("*.png")))
        self.dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        self.optimizer = Adam(self.model.parameters(), lr=2e-4)  

    def train(self):
        steps = 1
        for _ in range(100):
            for batch in self.dataloader:
                self.optimizer.zero_grad()
                images = batch
                images = images.to(DEVICE)

                loss= self.model(images)
                loss.backward()
                print(loss)
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