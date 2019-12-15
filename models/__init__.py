from models.vae import VAEModel
from .bert import BERTModel

MODELS = {
    BERTModel.code(): BERTModel,
    VAEModel.code(): VAEModel,
}


def model_factory(args, dataset_meta):
    model = MODELS[args.model_code]
    return model(args, dataset_meta)
