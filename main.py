from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *


def train():
    export_root = setup_train(args)
    meta, train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args, meta)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()
    # test_result = test_with(trainer.best_model, test_loader)
    # save_test_result(export_root, test_result)


def evaluate():
    export_root = setup_train(args)
    meta, train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args, meta)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)

    path = args.eval_model_path
    load_pretrained_weights(model, path)

    average_meter_set = AverageMeterSet()
    for batch in test_loader:
        with torch.no_grad():
            batch = [x.to(trainer.device) for x in batch]
            metrics = trainer.calculate_metrics(batch)
            for k, v in metrics.items():
                average_meter_set.update(k, v)
    print(average_meter_set.averages())


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'eval':
        evaluate()
    else:
        raise ValueError('Invalid mode')
