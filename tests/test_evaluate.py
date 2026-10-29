"""Tests for predict() in src/evaluate.py with a tiny linear model, so no GPU or real data is needed."""
import sys
from pathlib import Path

from PIL import Image
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import predict, ten_crop_transform

CPU = torch.device("cpu")


def tiny_model(num_classes=6):
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, num_classes))


def run(model, batches):
    return predict(model, batches, CPU, amp=False, memory_format=torch.contiguous_format)


def test_single_crop_matches_softmax_of_the_model():
    model = tiny_model()
    images, labels = torch.randn(5, 3, 4, 4), torch.tensor([0, 1, 2, 3, 4])
    pred = run(model, [(images, labels)])
    probs = model(images).softmax(dim=1)
    assert pred["top5"][:, 0].tolist() == probs.argmax(dim=1).tolist()
    assert pred["true_prob"] == pytest.approx(probs[range(5), labels].detach().numpy(), rel=1e-5)
    assert pred["loss"] == pytest.approx(-probs[range(5), labels].log().detach().numpy(), rel=1e-5)


def test_ten_crop_averages_the_probabilities_of_the_crops():
    model = tiny_model()
    a, b = torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)
    crops = torch.stack([a, b], dim=1)  # (2 images, 2 crops, 3, 4, 4)
    labels = torch.tensor([1, 3])
    pred = run(model, [(crops, labels)])
    expected = (model(a).softmax(dim=1) + model(b).softmax(dim=1)) / 2
    assert pred["true_prob"] == pytest.approx(expected[range(2), labels].detach().numpy(), rel=1e-5)
    assert pred["top5"][:, 0].tolist() == expected.argmax(dim=1).tolist()


def test_ten_crop_of_identical_crops_equals_single_crop():
    model = tiny_model()
    images, labels = torch.randn(3, 3, 4, 4), torch.tensor([0, 1, 2])
    single = run(model, [(images, labels)])
    ten = run(model, [(images[:, None].expand(-1, 10, -1, -1, -1), labels)])
    for key in single:
        assert ten[key] == pytest.approx(single[key], rel=1e-5)


def test_ten_crop_transform_gives_ten_normalized_crops():
    x = ten_crop_transform()(Image.new("RGB", (256, 256), (128, 64, 32)))
    assert x.shape == (10, 3, 224, 224)
