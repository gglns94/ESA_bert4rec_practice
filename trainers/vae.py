import numpy as np

from models.vae_modules.loss import annealed_elbo
from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks


class VAETrainer(AbstractTrainer):

    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.update_count = 0

    @classmethod
    def code(cls):
        return 'vae'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def backprop(self, batch):
        loss = super().backprop(batch)
        self.update_count += 1
        return loss

    def calculate_loss(self, batch):
        (data, ) = batch
        if self.args.vae_total_anneal_steps > 0:
            anneal = min(self.args.vae_anneal_cap, 1. * self.update_count / self.args.vae_total_anneal_steps)
        else:
            anneal = self.args.vae_anneal_cap

        recon_batch, mu, logvar = self.model(data)
        loss = annealed_elbo(recon_batch, data, mu, logvar, anneal)
        return loss

    def calculate_metrics(self, batch):
        data_tr, data_te = batch
        recon_batch, mu, logvar = self.model(data_tr)
        recon_batch[data_tr > 0] = -np.inf
        return recalls_and_ndcgs_for_ks(recon_batch, data_te, self.metric_ks)
