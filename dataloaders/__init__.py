from datasets import dataset_factory
from .bert import BertDataloader
from .vae import VAEDataloader


DATALOADERS = {
    BertDataloader.code(): BertDataloader,
    VAEDataloader.code(): VAEDataloader,
}


def dataloader_factory(args):
    dataset = dataset_factory(args)
    dataloader = DATALOADERS[args.dataloader_code]
    dataloader = dataloader(args, dataset)
    meta = dataloader.get_meta()
    train, val, test = dataloader.get_pytorch_dataloaders()
    return meta, train, val, test
