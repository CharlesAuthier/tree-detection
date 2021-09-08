from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


def create_model(hparams):

    config = get_efficientdet_config(hparams.architecture)
    config.update({'num_classes': hparams.num_classes})
    config.update({'image_size': (hparams.input_size, hparams.input_size)})

    # print(config)  # TODO make more options

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)

