#!/usr/bin/env python
# encoding: utf-8

from argparse import ArgumentParser
import torch
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import time

from speaker_trainer import Model, model_evaluation
from pytorch_lightning.callbacks import Callback

torch.multiprocessing.set_sharing_strategy('file_system')

def cli_main():
    # args
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Model(**vars(args))

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")["state_dict"]
        # pop loss Function parameter
        loss_weights = []
        for key, value in state_dict.items():
            if "loss" in key:
                loss_weights.append(key)
        for item in loss_weights:
            state_dict.pop(item)
        model.load_state_dict(state_dict, strict=False)
        print("initial parameter from pretrain model {}".format(args.checkpoint_path))

    if args.evaluate is not True:
        args.default_root_dir = "exp/" + args.nnet_type + "_" + args.pooling_type + "_" + args.loss_type + "_" + time.strftime('%Y-%m-%d-%H-%M-%S')
        checkpoint_callback = ModelCheckpoint(monitor='loss', save_top_k=args.save_top_k,
                filename="{epoch}_{train_loss:.2f}", dirpath=args.default_root_dir)
        args.checkpoint_callback = checkpoint_callback
        lr_monitor = LearningRateMonitor(logging_interval='step')
        args.callbacks = [model_evaluation(), lr_monitor]
        trainer = Trainer.from_argparse_args(args)
        trainer.fit(model)
    else:
        model.hparams.train_list_path = args.train_list_path
        model.cuda()
        model.eval()
        with torch.no_grad():
            model.cosine_evaluate()

if __name__ == '__main__':  # pragma: no cover
    cli_main()

