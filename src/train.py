from typing import List, Optional

import hydra
import torch
from effdet import DetBenchPredict
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

from effdet import get_efficientdet_config, EfficientDet
from effdet.efficientdet import HeadNet

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # Evaluate model on test set, using the best model achieved during training
    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Save the model in ONNX TODO ERROR and change place to be more polyvalant (now only effdet)
    # filepath = "model.onnx"
    # input_sample = torch.randn((1, 3, 256, 256))
    # checkpoint_path = trainer.checkpoint_callback.best_model_path
    # model = load_net(checkpoint_path, config.model)
    # export_onnx_model(model, input_sample, filepath)

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


def load_net(checkpoint_path, cfg):
    config = get_efficientdet_config(cfg.architecture)
    config.update({'num_classes': cfg.num_classes})
    config.update({'image_size': (cfg.input_size, cfg.input_size)})
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)

    checkpoint = torch.load(checkpoint_path)['state_dict']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        # name = k[7:]  # remove `module.`
        name = k.replace('model.', '')
        new_state_dict[name] = v
    new_state_dict = {k: v for k, v in new_state_dict.items() if 'anchors.boxes' not in k}

    net.load_state_dict(new_state_dict)

    net = DetBenchPredict(net)
    net.eval()
    return net


def export_onnx_model(model, input_sample, onnx_path, input_names=None, output_names=None, dynamic_axes=None):
    torch.onnx.export(
        model, input_sample, onnx_path,
        input_names=['img'], output_names=['output'],
        export_params=True,
        verbose=False,
        opset_version=11
    )