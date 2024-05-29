
import torch

class Dataset(torch.utils.data.Dataset):
    """
    Custom dataset class for handling data.

    Args:
        x (list): List of input data.
        radius_x (list): List of radius_x values.
        radius_y (list): List of radius_y values.
        center (list): List of center values.
        brightness (list): List of brightness values.
    """

    def __init__(self, x, radius, brightness):
        self.x = x
        self.radius = radius
        self.brightness = brightness

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the input data and the features.
        """
        x = self.x[idx][None]
        radius = self.radius[idx]
        brightness = self.brightness[idx]
        f = torch.stack((radius[0], radius[1], brightness), dim=0)
        return x, f