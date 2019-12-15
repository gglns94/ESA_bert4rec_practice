from models.base import BaseModel
from models.vae_modules.multi_vae import MultiVAE


class VAEModel(BaseModel):

    def __init__(self, args, dataset_meta):
        super().__init__(args, dataset_meta)
        self.multi_vae = MultiVAE(args.vae_p_dims + [dataset_meta['item_count']], args.vae_q_dims, dropout=args.vae_dropout)

    @classmethod
    def code(cls):
        return 'vae'

    def forward(self, x):
        x = self.multi_vae(x)
        return x
