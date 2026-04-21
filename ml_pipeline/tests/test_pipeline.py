# Tests for the ML pipeline model

import os
import sys
import tempfile
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import MNISTNet


def test_output_shape():
    model = MNISTNet()
    model.eval()
    out = model(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_single_image():
    model = MNISTNet()
    model.eval()
    out = model(torch.randn(1, 1, 28, 28))
    assert out.shape == (1, 10)


def test_output_sums_to_one():
    model = MNISTNet()
    model.eval()
    out = model(torch.randn(2, 1, 28, 28))
    probs = torch.exp(out)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_param_count():
    model = MNISTNet()
    total = sum(p.numel() for p in model.parameters())
    assert total < 1_000_000, f"Too many params: {total}"


def test_save_load():
    model = MNISTNet()
    model.eval()

    inp = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        original = model(inp)

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save({"model_state_dict": model.state_dict(), "arch_params": model.arch_params}, f.name)
        ckpt = torch.load(f.name, weights_only=True)

    loaded = MNISTNet(**ckpt["arch_params"])
    loaded.load_state_dict(ckpt["model_state_dict"])
    loaded.eval()

    with torch.no_grad():
        result = loaded(inp)

    assert torch.allclose(original, result, atol=1e-6)
    os.unlink(f.name)


def test_custom_architecture():
    model = MNISTNet(conv1_ch=8, conv2_ch=16, fc1_units=64, dropout=0.1)
    model.eval()
    out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)
    assert model.arch_params == {"conv1_ch": 8, "conv2_ch": 16, "fc1_units": 64, "dropout": 0.1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
