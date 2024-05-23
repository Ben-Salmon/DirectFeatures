from torch.distributions import Uniform, Normal
from PIL import Image, ImageDraw
import torch
import numpy as np

def generate_circle_image(radius, brightness, size=(16, 16)):
    """Generate a circle image with the given radius and brightness
    
    Args:
        radius (float): The radius of the circle
        brightness (0<float<255): The brightness of the circle
        size (tuple): The size of the image

    Returns:
        torch.Tensor: The circle image
    """
    # Create a blank image with black background
    image = Image.new("L", size, color=0)
    draw = ImageDraw.Draw(image)

    # Calculate the center coordinates
    center = (size[0]//2, size[1]//2)

    # Draw the circle
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        fill=int(brightness),
    )

    return torch.from_numpy(np.array(image)).float()

class P_Radius:
    """Probability distribution over cell radius"""
    def __init__(self, low, high):
        self.p = Uniform(low, high)

    def sample(self):
        return self.p.sample()

    def log_prob(self, f):
        return self.p.log_prob(f)
   

class P_Brightness:
    """Probability distribution over cell brightness"""
    def __init__(self, low, high):
        self.p = Uniform(low, high)

    def sample(self):
        return self.p.sample()

    def log_prob(self, f):
        return self.p.log_prob(f)
    
class P_X_under_F:
    """Probability distribution over the image given the cell radius and brightness
    
    Args:
        radius (float): The radius of the circle
        brightness (0<float<255): The brightness of the circle
        std (float): The standard deviation of the Gaussian noise
    """
    def __init__(self, radius, brightness, std):
        circle = generate_circle_image(radius, brightness)
        std = torch.Tensor([std])
        self.p = Normal(loc=circle, scale=std)

    def sample(self):
        return self.p.sample()

    def log_prob(self, x):
        return self.p.log_prob(x)
    

class P_F_under_X:
    """Unnormalised probability distribution over cell radius and brightness given the image

    Calculates posterior using prior over radius and brightness and likelihood of the image given the radius and brightness

    Args:
        p_x_under_f (P_X_under_F): Probability distribution over the image given the cell radius and brightness
        p_radius (P_Radius): Probability distribution over cell radius
        p_brightness (P_Brightness): Probability distribution over cell brightness
    """
    def __init__(self, p_x_under_f, p_radius, p_brightness):
        self.p_x_given_f = p_x_under_f
        self.p_radius = p_radius
        self.p_brightness = p_brightness

    def loglikelihood(self, x, radius, brightness):
        pradius = self.p_radius.log_prob(radius)
        pbrightness = self.p_brightness.log_prob(brightness)
        pf = pradius + pbrightness
        pxf = self.p_x_given_f.log_prob(x)
        pxf = torch.sum(pxf, dim=(-2, -1))
        return pf + pxf