import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *

def train_epoch(model, optimizer, train_loader, device):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        device (torch.device): The device to perform training on.

    Returns:
        float: The average loss for the epoch.
    """
    model = model.to(device)
    running_loss = 0
    for x_batch, f_batch in train_loader:
        x_batch, f_batch = x_batch.to(device), f_batch.to(device)
        optimizer.zero_grad()
        loss = model.loss(x_batch, f_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def val_epoch(model, val_loader, device):
    """
    Perform a validation epoch for the given model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: The average validation loss.
    """
    with torch.no_grad():
        model = model.to(device)
        running_loss = 0
        for x_batch, f_batch in val_loader:
            x_batch, f_batch = x_batch.to(device), f_batch.to(device)
            loss = model.loss(x_batch, f_batch)
            running_loss += loss.item()

        # Plot predicted posterior over each feature    
        x_test = x_batch[:1]
        pfx_test = model.get_p_f_under_x(x_test)

        radius_locs = pfx_test.component_distribution.loc[:, 0].cpu().numpy()
        radius_scales = pfx_test.component_distribution.scale[:, 0].cpu().numpy()
        radius_coeffs = pfx_test.mixture_distribution.probs[:, 0].cpu().numpy()

        brightness_locs = pfx_test.component_distribution.loc[:, 1].cpu().numpy()
        brightness_scales = pfx_test.component_distribution.scale[:, 1].cpu().numpy()
        brightness_coeffs = pfx_test.mixture_distribution.probs[:, 1].cpu().numpy()
        
        # Run subplots
        domains = []
        vis_data = []
        ymins, ymaxs = [], []
        titles = [] 
        #Radius
        ymin = 0
        ymax = 0.2
        domain = np.linspace(0, 8, 1000) # Possible radius values
        y_total = gen_gmm_pdf(radius_locs, radius_scales, radius_coeffs, n_gaussians=model.n_gaussians, domain=domain)
        vis_data.append(y_total)
        domains.append(domain)
        ymins.append(ymin)
        ymaxs.append(ymax)
        titles.append("Radius")
        #Brightness
        ymin = 0
        ymax = 0.02
        domain = np.linspace(0, 255, 1000) # Possible brightness values
        y_total = gen_gmm_pdf(brightness_locs, brightness_scales, brightness_coeffs, n_gaussians=model.n_gaussians, domain=domain)
        vis_data.append(y_total)
        domains.append(domain)
        ymins.append(ymin)
        ymaxs.append(ymax)
        titles.append("Brightness")
        #Create subplot
        create_subplot(1, len(vis_data), titles, vis_data, f_batch, domains, ymins, ymaxs)

        epoch_val_loss = running_loss / len(val_loader)
        print(f"Validation loss: {epoch_val_loss}")
        return epoch_val_loss


def train(model, optimizer, train_loader, val_loader, device, n_epochs):
    """
    Trains the model for a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for the training dataset.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation dataset.
        device (torch.device): The device to be used for training.
        n_epochs (int): The number of epochs to train the model.

    Returns:
        tuple: A tuple containing the training losses and validation losses for each epoch.
    """

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(n_epochs)):
        model.train()
        train_loss = train_epoch(model, optimizer, train_loader, device)
        model.eval()
        val_loss = val_epoch(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    return train_losses, val_losses