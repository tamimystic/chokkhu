import numpy as np
import chokkhu as ck
from chokkhu.core.tensor import Tensor
from chokkhu.models.vision import (
    CBAM,
    ChannelShuffle,
    ConvNeXtTiny,
    DenseNet121,
    EfficientNetB0,
    FCN8s,
    GoogLeNet,
    GroupedConv2D,
    InceptionV3,
    MobileNetV2,
    MobileNetV3,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNeXt50,
    ResNeXt101,
    SEBlock,
    ShuffleNetV2,
    SqueezeNet,
    VGG13,
    VGG19,
    VisionTransformer,
    ZFNet,
)


def test_attention_and_grouped_conv_layers():
    x = Tensor(np.random.randn(2, 16, 8, 8), requires_grad=True)

    # SEBlock
    se = SEBlock(channels=16, reduction=4)
    out_se = se(x)
    assert out_se.shape == (2, 16, 8, 8)
    out_se.sum().backward()
    assert x.grad is not None

    x.zero_grad()
    # CBAM
    cbam = CBAM(channels=16, reduction=4, kernel_size=3)
    out_cbam = cbam(x)
    assert out_cbam.shape == (2, 16, 8, 8)

    # GroupedConv2D
    gconv = GroupedConv2D(
        in_channels=16, out_channels=32, kernel_size=3, padding=1, groups=4
    )
    out_gc = gconv(x)
    assert out_gc.shape == (2, 32, 8, 8)

    # ChannelShuffle
    cs = ChannelShuffle(groups=4)
    out_cs = cs(out_gc)
    assert out_cs.shape == (2, 32, 8, 8)


def test_classic_and_vgg_variants():
    np.random.seed(42)
    X = np.random.randn(2, 3, 32, 32)

    # VGG13
    vgg13 = VGG13(num_classes=5, in_channels=3)
    preds13 = vgg13.predict(X)
    assert len(preds13) == 2

    # VGG19
    vgg19 = VGG19(num_classes=5, in_channels=3)
    preds19 = vgg19.predict(X)
    assert len(preds19) == 2

    # ZFNet (test with standard 224x224 input)
    zf = ZFNet(num_classes=5, in_channels=3)
    X_zf = np.random.randn(2, 3, 224, 224)
    preds_zf = zf.predict(X_zf)
    assert len(preds_zf) == 2


def test_inception_and_resnet_family():
    np.random.seed(42)
    X = np.random.randn(2, 3, 32, 32)

    # GoogLeNet
    googlenet = GoogLeNet(num_classes=5, in_channels=3)
    preds_g = googlenet.predict(X)
    assert len(preds_g) == 2

    # InceptionV3
    inc3 = InceptionV3(num_classes=5, in_channels=3)
    preds_inc3 = inc3.predict(X)
    assert len(preds_inc3) == 2

    # ResNet34
    r34 = ResNet34(num_classes=5, in_channels=3)
    preds_r34 = r34.predict(X)
    assert len(preds_r34) == 2

    # ResNet50
    r50 = ResNet50(num_classes=5, in_channels=3)
    preds_r50 = r50.predict(X)
    assert len(preds_r50) == 2

    # ResNet101
    r101 = ResNet101(num_classes=5, in_channels=3)
    preds_r101 = r101.predict(X)
    assert len(preds_r101) == 2

    # ResNet152
    r152 = ResNet152(num_classes=5, in_channels=3)
    preds_r152 = r152.predict(X)
    assert len(preds_r152) == 2


def test_resnext_and_densenet():
    np.random.seed(42)
    X = np.random.randn(2, 3, 32, 32)

    # ResNeXt50
    rx50 = ResNeXt50(num_classes=5, in_channels=3, cardinality=16)
    preds_rx50 = rx50.predict(X)
    assert len(preds_rx50) == 2

    # ResNeXt101
    rx101 = ResNeXt101(num_classes=5, in_channels=3, cardinality=16)
    preds_rx101 = rx101.predict(X)
    assert len(preds_rx101) == 2

    # DenseNet121
    dn121 = DenseNet121(num_classes=5, in_channels=3)
    preds_dn = dn121.predict(X)
    assert len(preds_dn) == 2

    # SqueezeNet
    sq = SqueezeNet(num_classes=5, in_channels=3)
    preds_sq = sq.predict(X)
    assert len(preds_sq) == 2


def test_mobile_and_modern_architectures():
    np.random.seed(42)
    X = np.random.randn(2, 3, 32, 32)

    # MobileNetV2
    mob2 = MobileNetV2(num_classes=5, in_channels=3)
    preds_mob2 = mob2.predict(X)
    assert len(preds_mob2) == 2

    # MobileNetV3
    mob3 = MobileNetV3(num_classes=5, in_channels=3)
    preds_mob3 = mob3.predict(X)
    assert len(preds_mob3) == 2

    # ShuffleNetV2
    shuf = ShuffleNetV2(num_classes=5, in_channels=3)
    preds_shuf = shuf.predict(X)
    assert len(preds_shuf) == 2

    # EfficientNetB0
    eff = EfficientNetB0(num_classes=5, in_channels=3)
    preds_eff = eff.predict(X)
    assert len(preds_eff) == 2

    # ConvNeXtTiny
    cnext = ConvNeXtTiny(num_classes=5, in_channels=3)
    preds_cnext = cnext.predict(X)
    assert len(preds_cnext) == 2


def test_fcn_and_vision_transformer():
    np.random.seed(42)
    X = np.random.randn(2, 3, 32, 32)

    # FCN8s
    fcn = FCN8s(num_classes=5, in_channels=3)
    out_fcn = fcn(Tensor(X))
    assert out_fcn.shape[0] == 2
    assert out_fcn.shape[1] == 5

    # VisionTransformer (ViT)
    vit = VisionTransformer(
        img_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=5,
        embed_dim=32,
        depth=2,
        num_heads=2,
    )
    preds_vit = vit.predict(X)
    assert len(preds_vit) == 2


def test_top_level_train_dispatching_for_all_vision_models():
    np.random.seed(42)
    X = np.random.randn(6, 3, 32, 32)
    y = np.random.choice([0, 1], size=6)

    # Test top-level train dispatchers
    m_eff = ck.train(
        model="efficientnet",
        X_train=X,
        y_train=y,
        epochs=1,
        batch_size=3,
        verbose=False,
    )
    assert m_eff is not None

    m_mob2 = ck.train(
        model="mobilenet_v2",
        X_train=X,
        y_train=y,
        epochs=1,
        batch_size=3,
        verbose=False,
    )
    assert m_mob2 is not None

    m_dense = ck.train(
        model="densenet", X_train=X, y_train=y, epochs=1, batch_size=3, verbose=False
    )
    assert m_dense is not None

    m_vit = ck.train(
        model="vit", X_train=X, y_train=y, epochs=1, batch_size=3, verbose=False
    )
    assert m_vit is not None
