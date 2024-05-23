
import torch

class Dataset(torch.utils.data.Dataset):
    """Dataset class for training

    Args:
        x (list): List of images
        radius (list): List of cell radii
        brightess (list): List of cell brightnesses
    """
    def __init__(self, x, radius, brightess):
        self.x = x
        self.radius = radius
        self.brightness = brightess

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """Stacks features into a single tensor along channel dimension
        """
        x = self.x[idx][None]
        radius = self.radius[idx]
        brightness = self.brightness[idx]
        f = torch.stack((radius, brightness), dim=0)
        return x, f