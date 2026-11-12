"""Focused tests for padding, the noise-channel direction, and reproducibility."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import HashTextCNN, ReviewModel, observed_probabilities, tokenize_hash, validate_transition
from train import load_checkpoint, save_checkpoint


def config(objective="forward", dense_dim=0):
    return {"objective": objective, "backbone": "hashcnn", "vocab_size": 128,
            "max_tokens": 16, "embedding_dim": 8, "cnn_channels": 4,
            "hidden_dim": 8, "dropout": 0., "dense_dim": dense_dim, "label_mean": 3.}


def test_hash_deterministic_and_empty():
    assert tokenize_hash("Good", "Not BAD!", 128, 16) == tokenize_hash("good", "not bad!", 128, 16)
    assert tokenize_hash("", "", 128, 16) == [0] * 16
    result = tokenize_hash("good", "food", 128, 16)
    assert result[0] == 1 and result[2] == 2 and len(result) == 16


def test_all_padding_is_zero_pool_and_finite_prediction():
    encoder = HashTextCNN(config())
    ids = torch.zeros((3, 16), dtype=torch.long)
    assert torch.equal(encoder(ids, ids), torch.zeros(3, 8))
    model = ReviewModel(config(), torch.eye(5))
    output = model(ids, ids)
    assert torch.isfinite(output).all()
    assert torch.isfinite(model.predictions(output)[0]).all()


def test_padding_does_not_change_pool():
    encoder = HashTextCNN(config()).eval()
    ids = torch.tensor([[1, 4, 5, 2, 6]])
    padded = torch.nn.functional.pad(ids, (0, 7))
    assert torch.allclose(encoder(ids, ids != 0), encoder(padded, padded != 0), atol=1e-6)


def test_transition_direction():
    # Latent one-star always observed as five-star. An asymmetric row disambiguates T vs T.T.
    transition = torch.eye(5)
    transition[0] = torch.tensor([0., 0., 0., 0., 1.])
    logits = torch.tensor([[50., -50., -50., -50., -50.]])
    observed = observed_probabilities(logits, transition)
    assert observed[0, 4] > .99999
    model = ReviewModel(config(), transition)
    observed_score, latent_score = model.predictions(logits)
    assert observed_score.item() == pytest.approx(5.)
    assert latent_score.item() == pytest.approx(1.)


def test_invalid_transition_rejected():
    with pytest.raises(ValueError):
        validate_transition(torch.ones(5, 5))
    with pytest.raises(ValueError):
        validate_transition(torch.zeros(4, 5))


@pytest.mark.parametrize("objective", ["mse", "ce", "forward"])
def test_loss_has_finite_nonzero_gradients(objective):
    transition = .8 * torch.eye(5) + .04 * torch.ones(5, 5)
    model = ReviewModel(config(objective, dense_dim=2), transition)
    ids = torch.tensor([tokenize_hash("great", "food", 128, 16), tokenize_hash("bad", "taste", 128, 16)])
    output = model(ids, ids != 0, torch.tensor([[.1, .2], [.2, .4]]))
    loss = model.loss(output, torch.tensor([5., 1.]))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.head[-1].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_identity_forward_loss_matches_ce():
    logits = torch.tensor([[1., 2., 3., 4., 5.], [2., 0., -1., 3., 1.]])
    target = torch.tensor([4., 2.])
    forward = ReviewModel(config("forward"), torch.eye(5))
    ce = ReviewModel(config("ce"), torch.eye(5))
    assert torch.allclose(forward.loss(logits, target), ce.loss(logits, target), atol=1e-6)


def test_checkpoint_roundtrip(tmp_path):
    model = ReviewModel(config(), .8 * torch.eye(5) + .04 * torch.ones(5, 5)).eval()
    ids = torch.tensor([tokenize_hash("Five Stars", "very nice", 128, 16)])
    before = model(ids, ids != 0).detach()
    checkpoint = tmp_path / "best.pt"
    save_checkpoint(checkpoint, model, epoch=2, cal_rmse=1.2, smoke=True)
    restored, payload = load_checkpoint(checkpoint)
    restored.eval()
    assert payload["epoch"] == 2 and payload["smoke"]
    assert torch.equal(before, restored(ids, ids != 0))
    assert torch.equal(model.transition, restored.transition)
