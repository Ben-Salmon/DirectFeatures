from torch.distributions import Uniform, Normal
from PIL import Image, ImageDraw
import torch
import numpy as np

def generate_circle_image(radius_x,radius_y,center, brightness, size):
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
    # center = (size[0]//2, size[1]//2)

    # Draw the circle
    draw.ellipse(
        (
            center[0] - radius_x,
            center[1] - radius_y,
            center[0] + radius_x,
            center[1] + radius_y,
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
   
class P_Center:

    def __init__(self, image_size, radius):
        up = image_size - radius[0]
        down = radius[0]
        left  = image_size - radius[1]
        right  = radius[1]
        
        self.p_x = Uniform(down, up)
        self.p_y = Uniform(right, left)

    def sample(self):
        return (self.p_x.sample(),self.p_y.sample())

    def log_prob(self, f):
        p_xy_log = self.p_x.log_prob(f[0]) + self.p_y.log_prob(f[1])
        return p_xy_log 


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
    def __init__(self, radius_x,radius_y, center, brightness, std, size):
        circle = generate_circle_image(radius_x,radius_y, center, brightness, size)
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
    def __init__(self, p_x_under_f, p_radius_x, p_radius_y, p_center, p_brightness):
        self.p_x_given_f = p_x_under_f
        self.p_radius_x = p_radius_x
        self.p_radius_y = p_radius_y
        self.p_brightness = p_brightness
        self.p_center = p_center

    def loglikelihood(self, x, radius_x, radius_y, center, brightness):
        pradius_x = self.p_radius_x.log_prob(radius_x)
        pradius_y = self.p_radius_y.log_prob(radius_y)
        pbrightness = self.p_brightness.log_prob(brightness)
        pcenter = self.p_center.log_prob(center)
        pf = pradius_x + pradius_y + pbrightness + pcenter
        pxf = self.p_x_given_f.log_prob(x)
        pxf = torch.sum(pxf, dim=(-2, -1))
        return pf + pxf