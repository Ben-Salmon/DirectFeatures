import numpy as np
import torch
import matplotlib.pyplot as plt

def gen_gaussian_pdf(loc, scale, x):
    """
    Generate the probability density function (PDF) of a Gaussian distribution.

    Parameters:
    loc (float): The mean (or location) of the Gaussian distribution.
    scale (float): The standard deviation (or scale) of the Gaussian distribution.
    x (array-like): The input values at which to evaluate the PDF.

    Returns:
    y (array-like): The PDF values corresponding to the input values.
    """
    loc = loc.squeeze()
    scale = scale.squeeze()
    y = np.exp(-0.5 * ((x - loc) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
    return y

def gen_gmm_pdf(locs, scales, coeffs, n_gaussians, domain):
    """
    Generate the probability density function (PDF) of a Gaussian Mixture Model (GMM).

    Parameters:
    locs (ndarray): The mean values of the Gaussian components. Shape: (..., n_gaussians).
    scales (ndarray): The standard deviations of the Gaussian components. Shape: (..., n_gaussians).
    coeffs (ndarray): The coefficients of the Gaussian components. Shape: (..., n_gaussians).
    n_gaussians (int): The number of Gaussian components in the GMM.
    domain (ndarray): The domain over which to evaluate the PDF.

    Returns:
    ndarray: The PDF of the GMM evaluated at the given domain.

    """
    y_total = np.zeros_like(domain)
    for i in range(n_gaussians):
        loc = locs[..., i]
        scale = scales[..., i]
        y = gen_gaussian_pdf(loc, scale, domain)
        y *= coeffs[..., i].squeeze()
        y_total += y
    return y_total


def create_subplot(rows, cols, titles, input_data, f_batch, domains, ymin, ymax):
    """
    Create subplots for the predicted posterior over each feature
    
    Parameters:
    - rows (int): Number of rows in the subplot grid.
    - cols (int): Number of columns in the subplot grid.
    - titles (list): List of titles for each subplot.
    - input_data (list): List of input data for each subplot.
    - f_batch (torch.Tensor): Tensor containing the predicted posterior for each feature.
    - domains (list): List of domains for each subplot.
    - ymin (list): List of minimum y-values for each subplot.
    - ymax (list): List of maximum y-values for each subplot.
    """
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(domains[i], input_data[i])
        ax.vlines(f_batch[0, i].item(), ymin=ymin[i], ymax=ymax[i], color="red")
        ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()

   
