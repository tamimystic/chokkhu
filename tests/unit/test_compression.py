"""Unit tests for Edge AI, Quantization & Model Compression Module."""

import numpy as np
import pytest

from chokkhu.compression import (
    UniformQuantizer,
    PostTrainingQuantizer,
    QuantizedLinear,
    MagnitudePruner,
    GlobalMagnitudePruner,
    KnowledgeDistiller,
    FeatureDistiller,
)


def test_uniform_quantizer_symmetric() -> None:
    np.random.seed(42)
    x = np.random.uniform(-5.0, 5.0, size=(10, 10)).astype(np.float32)

    q = UniformQuantizer(bits=8, symmetric=True)
    q_arr = q.quantize(x)

    assert q_arr.dtype == np.int8
    assert np.all(q_arr >= -128)
    assert np.all(q_arr <= 127)

    x_rec = q.dequantize(q_arr)
    # Quantization error for 8-bit over [-5, 5] is roughly 10 / 256 ≈ 0.04
    max_err = float(np.max(np.abs(x - x_rec)))
    assert max_err < 0.1

    # Fake quantize
    fq = q.fake_quantize(x)
    assert np.allclose(fq, x_rec, atol=1e-5)


def test_uniform_quantizer_asymmetric() -> None:
    np.random.seed(42)
    # Asymmetric input in [0, 10]
    x = np.random.uniform(0.0, 10.0, size=(20, 20)).astype(np.float32)

    q = UniformQuantizer(bits=8, symmetric=False)
    q_arr = q.quantize(x)

    assert q_arr.dtype == np.uint8
    assert np.all(q_arr >= 0)
    assert np.all(q_arr <= 255)

    x_rec = q.dequantize(q_arr)
    max_err = float(np.max(np.abs(x - x_rec)))
    assert max_err < 0.1


def test_uniform_quantizer_int4() -> None:
    np.random.seed(42)
    x = np.random.uniform(-3.0, 3.0, size=(15, 15)).astype(np.float32)

    q = UniformQuantizer(bits=4, symmetric=True)
    q_arr = q.quantize(x)

    assert np.all(q_arr >= -8)
    assert np.all(q_arr <= 7)

    x_rec = q.dequantize(q_arr)
    # 4-bit has 16 bins -> max error is higher but bounded
    max_err = float(np.max(np.abs(x - x_rec)))
    assert max_err < 1.0


def test_post_training_quantizer() -> None:
    np.random.seed(42)
    calib_data = np.random.normal(0.0, 2.0, size=(100, 50)).astype(np.float32)

    # MinMax calibration
    ptq_minmax = PostTrainingQuantizer(method="minmax", bits=8)
    q_minmax = ptq_minmax.calibrate(calib_data)
    assert q_minmax.is_calibrated

    # Percentile calibration
    ptq_perc = PostTrainingQuantizer(method="percentile", bits=8, percentile=99.0)
    q_perc = ptq_perc.calibrate(calib_data)
    assert q_perc.is_calibrated
    assert q_perc.scale <= q_minmax.scale

    # KL Divergence calibration
    ptq_kl = PostTrainingQuantizer(method="kl_divergence", bits=8)
    q_kl = ptq_kl.calibrate(calib_data)
    assert q_kl.is_calibrated


def test_quantized_linear() -> None:
    np.random.seed(42)
    in_dim = 16
    out_dim = 8
    batch_size = 4

    weight = np.random.randn(out_dim, in_dim).astype(np.float32)
    bias = np.random.randn(out_dim).astype(np.float32)
    x = np.random.randn(batch_size, in_dim).astype(np.float32)

    # Reference float forward
    fp32_out = x @ weight.T + bias

    # Quantized linear
    q_linear = QuantizedLinear(in_features=in_dim, out_features=out_dim)
    q_linear.quantize_from_float(weight, bias)
    q_out = q_linear(x)

    assert q_out.shape == (batch_size, out_dim)
    # Output should be close to FP32 output
    cos_sim = float(
        np.sum(fp32_out * q_out) / (np.linalg.norm(fp32_out) * np.linalg.norm(q_out))
    )
    assert cos_sim > 0.99


def test_magnitude_pruner_unstructured() -> None:
    np.random.seed(42)
    w = np.random.randn(20, 20).astype(np.float32)

    pruner = MagnitudePruner(amount=0.5, structured=False)
    pruned_w, mask = pruner.prune(w)

    assert pruned_w.shape == w.shape
    assert mask.shape == w.shape
    sparsity = pruner.compute_sparsity(pruned_w)
    assert pytest.approx(sparsity, 0.05) == 0.5

    # Zero weights should match mask == 0
    assert np.all((pruned_w == 0.0) == (mask == 0.0))


def test_magnitude_pruner_structured() -> None:
    np.random.seed(42)
    w = np.random.randn(10, 20).astype(np.float32)

    # Prune 40% of rows (output channels)
    pruner = MagnitudePruner(amount=0.4, structured=True, dim=0)
    pruned_w, mask = pruner.prune(w)

    # Check that entire rows are zeroed out
    row_sums = np.sum(np.abs(pruned_w), axis=1)
    zero_rows = int(np.sum(row_sums == 0.0))
    assert zero_rows == 4  # 40% of 10 rows


def test_global_magnitude_pruner() -> None:
    np.random.seed(42)
    weights_dict = {
        "w1": np.random.randn(10, 10).astype(np.float32),
        "w2": np.random.randn(20, 10).astype(np.float32),
    }

    global_pruner = GlobalMagnitudePruner(amount=0.3)
    pruned_dict, masks_dict = global_pruner.prune_dict(weights_dict)

    total_elements = sum(v.size for v in weights_dict.values())
    total_zeros = sum(int(np.sum(v == 0.0)) for v in pruned_dict.values())
    global_sparsity = float(total_zeros) / float(total_elements)

    assert pytest.approx(global_sparsity, 0.05) == 0.3


def test_knowledge_distiller_classification() -> None:
    np.random.seed(42)
    batch_size = 8
    num_classes = 5

    student_logits = np.random.randn(batch_size, num_classes).astype(np.float32)
    teacher_logits = np.random.randn(batch_size, num_classes).astype(np.float32)
    targets = np.random.randint(0, num_classes, size=batch_size)

    distiller = KnowledgeDistiller(
        temperature=2.0, alpha=0.5, task_type="classification"
    )
    total_loss, task_loss, distill_loss = distiller.compute_loss(
        student_logits, teacher_logits, targets
    )

    assert total_loss > 0.0
    assert task_loss > 0.0
    assert distill_loss > 0.0
    assert pytest.approx(total_loss, 1e-4) == 0.5 * task_loss + 0.5 * distill_loss


def test_knowledge_distiller_regression() -> None:
    np.random.seed(42)
    batch_size = 10
    student_preds = np.random.randn(batch_size, 1).astype(np.float32)
    teacher_preds = np.random.randn(batch_size, 1).astype(np.float32)
    y_true = np.random.randn(batch_size, 1).astype(np.float32)

    distiller = KnowledgeDistiller(alpha=0.3, task_type="regression")
    total_loss, task_loss, distill_loss = distiller.compute_loss(
        student_preds, teacher_preds, y_true
    )

    assert total_loss > 0.0
    assert pytest.approx(total_loss, 1e-4) == 0.7 * task_loss + 0.3 * distill_loss


def test_feature_distiller() -> None:
    np.random.seed(42)
    batch_size = 4
    student_feat = np.random.randn(batch_size, 32).astype(np.float32)
    teacher_feat = np.random.randn(batch_size, 64).astype(np.float32)

    feat_distiller = FeatureDistiller(student_dim=32, teacher_dim=64)
    loss = feat_distiller.compute_feature_loss(student_feat, teacher_feat)

    assert loss > 0.0
    assert isinstance(loss, float)


def test_compression_exceptions() -> None:
    with pytest.raises(ValueError):
        UniformQuantizer(bits=1)  # bits < 2

    with pytest.raises(ValueError):
        MagnitudePruner(amount=1.5)  # amount >= 1.0

    with pytest.raises(ValueError):
        KnowledgeDistiller(temperature=-1.0)

    with pytest.raises(ValueError):
        KnowledgeDistiller(task_type="unsupported")
