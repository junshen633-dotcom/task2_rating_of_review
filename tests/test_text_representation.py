"""Representation contracts: identical tokens, isolated streams, masked pools."""
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import HashTextCNN, ReviewModel, tokenize_hash


def config(layout="joint", pooling="mean"):
    return {"objective": "forward", "backbone": "hashcnn", "vocab_size": 64,
            "max_tokens": 16, "embedding_dim": 4, "cnn_channels": 3,
            "kernels": [3, 5], "hidden_dim": 8, "dropout": 0., "dense_dim": 2,
            "text_layout": layout, "pooling": pooling}


@pytest.mark.parametrize("layout", ["joint", "separate"])
@pytest.mark.parametrize("pooling", ["mean", "mean_max"])
def test_all_ablation_shapes_empty_padding_and_gradients(layout, pooling):
    cfg = config(layout, pooling)
    encoder = HashTextCNN(cfg).eval()
    ids = torch.tensor([tokenize_hash("Good", "not bad at all", 64, 16), [0]*16])
    actual = encoder(ids, ids != 0)
    width = 6 * (2 if layout == "separate" else 1) * (2 if pooling == "mean_max" else 1)
    assert actual.shape == (2, width)
    assert torch.equal(actual[1], torch.zeros(width))
    padded = F.pad(ids, (0, 8))
    assert torch.allclose(actual, encoder(padded, padded != 0), atol=1e-6)
    model = ReviewModel(cfg, .75*torch.eye(5)+.05*torch.ones(5, 5))
    loss = model.loss(model(ids, ids != 0, torch.zeros(2, 2)), torch.tensor([4., 1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_legacy_default_is_bitwise_equal_and_loads_identical_state_keys():
    cfg = config()
    cfg.pop("text_layout")
    cfg.pop("pooling")
    torch.manual_seed(12)
    legacy = HashTextCNN(cfg).eval()
    explicit = HashTextCNN(config()).eval()
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    ids = torch.tensor([tokenize_hash("Fine", "this is very good", 64, 16), [0]*16])
    mask = (ids != 0).to(legacy.embedding.weight.dtype).unsqueeze(1)
    embedded = legacy.embedding(ids).transpose(1, 2)*mask
    expected = torch.cat([(F.relu(conv(embedded))*mask).sum(-1)/mask.sum(-1).clamp_min(1)
                          for conv in legacy.convolutions], dim=1)
    assert torch.equal(legacy(ids, ids != 0), expected)
    assert torch.equal(explicit(ids, ids != 0), expected)
    assert list(legacy.state_dict()) == ["embedding.weight", "convolutions.0.weight",
                                       "convolutions.0.bias", "convolutions.1.weight", "convolutions.1.bias"]


@pytest.mark.parametrize("pooling", ["mean", "mean_max"])
def test_separate_stream_cannot_see_other_field_even_when_title_length_changes(pooling):
    cfg = config("separate", pooling)
    cfg.update(embedding_dim=1, cnn_channels=1, kernels=[3])
    encoder = HashTextCNN(cfg).eval()
    with torch.no_grad():
        encoder.embedding.weight[:, 0] = torch.arange(64).float()/10
        encoder.convolutions[0].weight.fill_(1)
        encoder.convolutions[0].bias.fill_(.1)
    rows = [[1, 3, 4, 2, 5, 6], [1, 7, 8, 2, 5, 6],
            [1, 7, 8, 9, 2, 5, 6], [1, 3, 4, 2, 11, 12]]
    ids = torch.tensor([row+[0]*(16-len(row)) for row in rows])
    output = encoder(ids, ids != 0)
    half = encoder.output_dim//2
    # Body is unchanged despite new title words and a shifted COMMENT boundary.
    assert torch.allclose(output[0, half:], output[1, half:], atol=1e-6)
    assert torch.allclose(output[0, half:], output[2, half:], atol=1e-6)
    assert not torch.equal(output[0, :half], output[1, :half])
    # Changing the body cannot alter title features.
    assert torch.allclose(output[0, :half], output[3, :half], atol=1e-6)
    assert not torch.equal(output[0, half:], output[3, half:])


def test_mean_max_uses_only_active_positions_and_has_no_empty_sentinel():
    cfg = config("joint", "mean_max")
    cfg.update(embedding_dim=1, cnn_channels=1, kernels=[1])
    encoder = HashTextCNN(cfg)
    with torch.no_grad():
        encoder.embedding.weight.zero_()
        encoder.embedding.weight[3, 0] = 2
        encoder.embedding.weight[4, 0] = 5
        encoder.embedding.weight[7, 0] = 10000
        encoder.convolutions[0].weight.fill_(1)
        encoder.convolutions[0].bias.fill_(1)
    ids = torch.tensor([[3, 4, 7], [3, 4, 7]])
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    assert torch.equal(encoder(ids, mask), torch.tensor([[4.5, 6.], [0., 0.]]))


@pytest.mark.parametrize("row", [[1, 3, 4, 0], [1, 2, 2, 0], [3, 2, 0, 0], [1, 1, 2, 0]])
def test_separate_rejects_missing_or_ambiguous_markers(row):
    encoder = HashTextCNN(config("separate"))
    ids = torch.tensor([row])
    with pytest.raises(ValueError, match="TITLE=1"):
        encoder(ids, ids != 0)


@pytest.mark.parametrize("layout,pooling", [("invalid", "mean"), ("joint", "invalid")])
def test_unknown_ablation_configuration_rejected(layout, pooling):
    with pytest.raises(ValueError):
        HashTextCNN(config(layout, pooling))
