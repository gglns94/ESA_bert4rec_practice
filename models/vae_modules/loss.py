import torch
import torch.nn.functional as F


def annealed_elbo(recon_x, x, mu, logvar, anneal):
    cross_entropy = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    kl_divergence = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return cross_entropy + anneal * kl_divergence
