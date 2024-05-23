import torch.nn as nn
from torch.distributions import Normal, Uniform, Categorical, MixtureSameFamily

class CNN(nn.Module):
    """Convolutional Neural Network for estimating fluorescence features from the image

    Args:
        n_filters (int): The number of filters in the convolutional layers
        x_mean (float): The mean of the input data
        x_std (float): The standard deviation of the input data
        img_size (tuple): The size of the image
        n_gaussians (int): The number of Gaussian components in the mixture model for each feature
        n_features (int): The number of fluorescence features in the output
    """
    def __init__(self, n_filters, x_mean, x_std, img_size=(16, 16), n_gaussians=1, n_features=2):
        super(CNN, self).__init__()
        self.x_mean = x_mean
        self.x_std = x_std
        self.n_gaussians = n_gaussians

        out_channels = 3 * n_gaussians * n_features

        self.n_filters = n_filters
        self.img_size = img_size
        self.conv1 = nn.Conv2d(1, n_filters, 9, padding=4)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 9, padding=4)
        self.conv3 = nn.Conv2d(n_filters, n_filters, 9, padding=4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(img_size[0] * img_size[1] * n_filters, out_channels)

    def forward(self, x):
        x = (x - self.x_mean) / self.x_std
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, self.img_size[0] * self.img_size[1] * self.n_filters)
        x = self.fc(x)
        return x

    def get_gaussian_params(self, params):
        logweights = params[:, 0::3].unfold(1, self.n_gaussians, self.n_gaussians)
        loc = params[:, 1::3].unfold(1, self.n_gaussians, self.n_gaussians)
        scale = params[:, 2::3].unfold(1, self.n_gaussians, self.n_gaussians)
        scale = nn.functional.softplus(scale)
        return logweights, loc, scale

    def get_p_f_under_x(self, x):
        params = self.forward(x)
        logweights, means, stds = self.get_gaussian_params(params)
        p = MixtureSameFamily(
            Categorical(logits=logweights, validate_args=True),
            Normal(loc=means, scale=stds, validate_args=True),
        )
        return p

    def loss(self, x, f):
        p = self.get_p_f_under_x(x)
        return -p.log_prob(f).mean()