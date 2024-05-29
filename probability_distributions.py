from torch.distributions import Uniform, Normal, Categorical
from PIL import Image, ImageDraw
import torch
import numpy as np
from tqdm import tqdm


def generate_circle_image(radius, brightness, img_size):
    """Generate a circle image with the given radius and brightness

    Args:
        radius (float): The radius of the circle
        brightness (0<float<255): The brightness of the circle
        size (tuple): The size of the image

    Returns:
        torch.Tensor: The circle image
    """
    # Create a blank image with black background
    image = Image.new("L", img_size, color=0)
    draw = ImageDraw.Draw(image)

    # Calculate the center coordinates
    center = (img_size[0] // 2, img_size[1] // 2)

    # Draw the circle
    draw.ellipse(
        (
            center[0] - radius[0],
            center[1] - radius[1],
            center[0] + radius[0],
            center[1] + radius[1],
        ),
        fill=int(brightness),
    )

    return torch.from_numpy(np.array(image)).float()


class P_Radius:
    """Probability distribution over cell radius"""

    def __init__(self, high):
        self.high = high
        probs = torch.ones(high) / high
        self.p = Categorical(probs)

    def sample(self):
        return (self.p.sample(), self.p.sample())

    def log_prob(self, f):
        return self.p.log_prob(f[0]) + self.p.log_prob(f[1])


class P_Center:

    def __init__(self, image_size, radius):
        up = image_size - radius[0]
        down = radius[0]
        left = image_size - radius[1]
        right = radius[1]

        self.p_x = Uniform(down, up)
        self.p_y = Uniform(right, left)

    def sample(self):
        return (self.p_x.sample(), self.p_y.sample())

    def log_prob(self, f):
        p_xy_log = self.p_x.log_prob(f[0]) + self.p_y.log_prob(f[1])
        return p_xy_log


class P_Brightness:
    """Probability distribution over cell brightness"""

    def __init__(self, high):
        self.high = high
        probs = torch.ones(high) / high
        self.p = Categorical(probs)

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

    def __init__(self, radius, brightness, std, size):
        self.size = size
        circle = generate_circle_image(radius, brightness, size)
        self.std = torch.Tensor([std])
        self.p = Normal(loc=circle, scale=self.std)

    def sample(self):
        return self.p.sample()

    def log_prob(self, x):
        return self.p.log_prob(x).sum(dim=(-2, -1))


class P_F_under_X:
    """Nnormalised probability distribution over cell radius and brightness given the image

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

    def sum_over_f(self, x, P_X_under_F, p_radius, p_brightness, std, size):
        px = 0
        for b in tqdm(range(p_brightness.high)):
            for rx in range(p_radius.high):
                for ry in range(p_radius.high):
                    b = torch.tensor([b])
                    rx = torch.tensor([rx])
                    ry = torch.tensor([ry])
                    log_pradius = p_radius.log_prob((rx, ry))
                    log_pbrightness = p_brightness.log_prob(b)
                    log_pf = log_pradius + log_pbrightness
                    p_x_under_f = P_X_under_F((rx, ry), b, std, size)
                    log_px_under_f = p_x_under_f.log_prob(x)
                    px += (log_pf + log_px_under_f).exp()
                    if (log_pf + log_px_under_f).exp() != 0:
                        print((log_pf + log_px_under_f).exp())
        return px

    def loglikelihood(self, x, radius, brightness):
        log_pradius = self.p_radius.log_prob(radius)
        log_pbrightness = self.p_brightness.log_prob(brightness)
        log_pf = log_pradius + log_pbrightness
        log_pxf = self.p_x_given_f.log_prob(x)

        px = self.sum_over_f(
            x,
            P_X_under_F=P_X_under_F,
            p_radius=self.p_radius,
            p_brightness=self.p_brightness,
            std=self.p_x_given_f.std,
            size=self.p_x_given_f.size,
        )
        log_px = torch.log(px)
        log_p_x_under_f = log_pf + log_pxf - log_px
        return log_p_x_under_f
