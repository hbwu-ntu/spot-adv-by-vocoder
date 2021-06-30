#!/usr/bin/env python
# coding=utf-8

from pytorch_lightning.callbacks import Callback
import torch

class model_evaluation(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if pl_module.hparams.eval_interval > 0 and epoch % pl_module.hparams.eval_interval == 0:
            pl_module.eval()
            with torch.no_grad():
                pl_module.cosine_evaluate()
        pl_module.train()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pass

