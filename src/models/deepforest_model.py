from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from src.models.modules.effdet_create_model import create_model


class DeepForestLitModel(LightningModule):
    """
    Example of LightningModule for DeepForest tree detection.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        architecture: str = 'efficientdet_d0',
        input_size: int = 256,
        num_classes: int = 2,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.model = create_model(hparams=self.hparams)

        # loss function
        # self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.model(x, y)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x, y)
        loss = {
            "loss": logits["loss"],
            "class_loss": logits["class_loss"],
            "box_loss": logits["box_loss"],
        }
        # preds = logits["detections"]
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        self.log("train/loss", loss['loss'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/class_loss", loss['class_loss'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/box_loss", loss['box_loss'], on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return loss['loss']

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        self.log("val/loss", loss['loss'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/class_loss", loss['class_loss'], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/box_loss", loss['box_loss'], on_step=False, on_epoch=True, prog_bar=False)

        return loss['loss']

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)

        self.log("test/loss", loss['loss'], on_step=False, on_epoch=True)
        self.log("test/class_loss", loss['class_loss'], on_step=False, on_epoch=True)
        self.log("test/box_loss", loss['box_loss'], on_step=False, on_epoch=True)

        return loss['loss']

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.AdamW(
            params=self.parameters(), lr=self.hparams.lr
        )
