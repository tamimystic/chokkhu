from __future__ import annotations

from .alexnet import AlexNet, ZFNet
from .convnext import ConvNeXt, ConvNeXtBlock, ConvNeXtTiny
from .deeplab import ASPPModule, DeepLabV3Plus
from .deit import DeiT, DeiTBase, DeiTSmall, DeiTTiny
from .densenet import DenseBlock, DenseNet121, DenseNet169, DenseNet201, TransitionBlock
from .efficientnet import EfficientNetB0, MBConv
from .fcn import FCN, FCN8s
from .inception import GoogLeNet, InceptionModule, InceptionV1, InceptionV3
from .lenet import LeNet5
from .mobilenet import InvertedResidual, MobileNetV1, MobileNetV2, MobileNetV3
from .resnet import (
    BottleneckResidualBlock,
    ResidualBlock,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
)
from .resnext import ResNeXt50, ResNeXt101
from .sam import SAMTwoWayAttentionBlock, SegmentAnythingModel
from .shufflenet import ChannelShuffle, ShuffleNetV1, ShuffleNetV2
from .squeezenet import FireModule, SqueezeNet
from .swin import SwinBase, SwinSmall, SwinT, SwinTransformer
from .unet import UNet
from .vgg import VGG11, VGG13, VGG16, VGG19
from .vit import PatchEmbedding, VisionTransformer, ViT, ViTBase, ViTTiny

__all__ = [
    "LeNet5",
    "AlexNet",
    "ZFNet",
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
    "GoogLeNet",
    "InceptionModule",
    "InceptionV1",
    "InceptionV3",
    "ResidualBlock",
    "BottleneckResidualBlock",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNeXt50",
    "ResNeXt101",
    "DenseBlock",
    "TransitionBlock",
    "DenseNet121",
    "DenseNet169",
    "DenseNet201",
    "FireModule",
    "SqueezeNet",
    "MobileNetV1",
    "MobileNetV2",
    "MobileNetV3",
    "InvertedResidual",
    "ChannelShuffle",
    "ShuffleNetV1",
    "ShuffleNetV2",
    "EfficientNetB0",
    "MBConv",
    "ConvNeXtBlock",
    "ConvNeXtTiny",
    "ConvNeXt",
    "UNet",
    "FCN",
    "FCN8s",
    "DeepLabV3Plus",
    "ASPPModule",
    "SegmentAnythingModel",
    "SAMTwoWayAttentionBlock",
    "PatchEmbedding",
    "VisionTransformer",
    "ViT",
    "ViTTiny",
    "ViTBase",
    "SwinTransformer",
    "SwinT",
    "SwinSmall",
    "SwinBase",
    "DeiT",
    "DeiTTiny",
    "DeiTSmall",
    "DeiTBase",
]
