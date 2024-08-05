from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from torchvision.transforms import ToTensor, Resize, Compose

transform = Compose(
   [Resize((512,512),antialias=True),
    ToTensor()]
)

class SwissRollDataset(Dataset):
    def __init__(self, image_paths, transform=transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('L')  # convert image to grayscale
        if self.transform:
            image = self.transform(image)
        return image
    

if __name__ == "__main__":
    rootdir = Path(f"{Path(__file__).parents[0]}/dummy_data")
    image_paths = list(rootdir.rglob("*.png"))
    dataset = SwissRollDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
