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
<<<<<<< HEAD
        torch.manual_seed(412)
        self.model = Diffusion(UNetWithTime()).to(DEVICE)
=======
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Diffusion(UNetWithTime()).to(self.device)
        # self.model = Diffusion(UNet(1)).to(self.device)
>>>>>>> 775d500 (fix: activation)
        dataset = SwissRollDataset(list(Path(f"{Path(__file__).parents[0]}/data/dummy_data").rglob("*.png")))
        self.dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
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