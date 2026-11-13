"""Small, auditable review regressors and observed-label noise likelihoods.

The transition matrix is T[latent_star - 1, observed_rating - 1].
Title stars are only a proxy for estimating T; the model does not assume that
this estimate proves a particular label-corruption mechanism.
"""
from __future__ import annotations

import re
import zlib
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
STARS = (1., 2., 3., 4., 5.)


def tokenize_hash(title: str, comment: str, vocab_size: int, max_tokens: int) -> list[int]:
    """Stable CRC32 hashing; PAD=0, TITLE=1, COMMENT=2, lexical IDs >=3.

    Reserve at most a quarter of the sequence for the title. Empty reviews
    deliberately remain all padding so empty-input behavior is testable.
    """
    if vocab_size < 8 or max_tokens < 8:
        raise ValueError("vocab_size and max_tokens must be at least 8")
    title_tokens = TOKEN_RE.findall(str(title or "").lower())
    comment_tokens = TOKEN_RE.findall(str(comment or "").lower())
    if not title_tokens and not comment_tokens:
        return [0] * max_tokens
    title_tokens = title_tokens[:min(32, max_tokens // 4)]
    comment_tokens = comment_tokens[:max_tokens - len(title_tokens) - 2]
    hashed = lambda word: 3 + zlib.crc32(word.encode("utf-8")) % (vocab_size - 3)
    ids = [1] + [hashed(t) for t in title_tokens]
    ids += [2] + [hashed(t) for t in comment_tokens]
    return ids + [0] * (max_tokens - len(ids))


def validate_transition(transition: Any) -> torch.Tensor:
    matrix = torch.as_tensor(transition, dtype=torch.float32)
    if matrix.shape != (5, 5):
        raise ValueError("transition must have shape (5, 5): latent rows, observed columns")
    if not torch.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("transition must contain finite, nonnegative probabilities")
    if not torch.allclose(matrix.sum(1), torch.ones(5), atol=1e-4, rtol=0):
        raise ValueError("every transition ROW must sum to 1")
    return matrix


def observed_probabilities(logits: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
    """Apply the forward channel in float32, including under mixed precision."""
    return logits.float().softmax(-1) @ transition.float()


class HashTextCNN(nn.Module):
    """Shared CNN with independent field encoding and masked pooling ablations.

    The default joint/mean path is the original implementation. Separate mode
    uses the very same retained token sequence and CNN weights, but zeroes the
    other field BEFORE convolution and pools each field independently. Thus a
    title/body boundary cannot create a cross-field n-gram.
    """
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        dim = int(config.get("embedding_dim", 64))
        channels = int(config.get("cnn_channels", 64))
        kernels = config.get("kernels", [3, 5])
        if not kernels or any(k < 1 or k % 2 == 0 for k in kernels):
            raise ValueError("CNN kernels must be positive odd integers")
        self.text_layout = config.get("text_layout", "joint")
        self.pooling = config.get("pooling", "mean")
        if self.text_layout not in ("joint", "separate"):
            raise ValueError("text_layout must be joint or separate")
        if self.pooling not in ("mean", "mean_max"):
            raise ValueError("pooling must be mean or mean_max")
        self.embedding = nn.Embedding(config["vocab_size"], dim, padding_idx=0)
        self.convolutions = nn.ModuleList(nn.Conv1d(dim, channels, k, padding=k // 2)
                                          for k in kernels)
        self.output_dim = (channels * len(kernels)
                           * (2 if self.pooling == "mean_max" else 1)
                           * (2 if self.text_layout == "separate" else 1))

    def _pool(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denominator = mask.sum(-1).clamp_min(1)
        pooled = []
        for conv in self.convolutions:
            activations = F.relu(conv(embedded)) * mask
            pooled.append(activations.sum(-1) / denominator)
            if self.pooling == "mean_max":
                # ReLU outputs are nonnegative, so zero is a safe masked value
                # for max pooling. Empty streams return zero without sentinels.
                pooled.append(activations.amax(-1))
        return torch.cat(pooled, dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(self.embedding.weight.dtype).unsqueeze(1)
        embedded = self.embedding(input_ids).transpose(1, 2) * mask
        if self.text_layout == "joint" and self.pooling == "mean":
            denominator = mask.sum(-1).clamp_min(1)
            # Keep legacy operations and parameter names unchanged, allowing
            # existing checkpoints and exact default predictions to reproduce.
            return torch.cat([(F.relu(conv(embedded)) * mask).sum(-1) / denominator
                              for conv in self.convolutions], dim=1)
        if self.text_layout == "joint":
            return self._pool(embedded, mask)

        active = attention_mask.bool()
        nonempty = active.any(-1)
        separator = (input_ids == 2) & active
        title_marker = (input_ids == 1) & active
        malformed = nonempty & ((separator.sum(-1) != 1) | (title_marker.sum(-1) != 1)
                                | (input_ids[:, 0] != 1) | ~active[:, 0])
        if malformed.any():
            raise ValueError("Separate layout requires one leading TITLE=1 and one COMMENT=2 per nonempty row")
        body_active = active & (separator.cumsum(-1) > 0)
        title_active = active & ~body_active
        title_mask = title_active.to(mask.dtype).unsqueeze(1)
        body_mask = body_active.to(mask.dtype).unsqueeze(1)
        return torch.cat((self._pool(embedded * title_mask, title_mask),
                          self._pool(embedded * body_mask, body_mask)), dim=1)


class ReviewModel(nn.Module):
    def __init__(self, config: dict[str, Any], transition: Any, *, pretrained: bool = True):
        super().__init__()
        self.config = dict(config)
        self.objective = config["objective"]
        if self.objective not in ("mse", "ce", "forward"):
            raise ValueError("objective must be mse, ce, or forward")
        self.register_buffer("transition", validate_transition(transition))
        self.register_buffer("stars", torch.tensor(STARS))
        if config["backbone"] == "hashcnn":
            self.encoder = HashTextCNN(config)
            text_dim = self.encoder.output_dim
        elif config["backbone"] == "hf":
            from transformers import AutoConfig, AutoModel
            if pretrained:
                self.encoder = AutoModel.from_pretrained(
                    config["model_name"], trust_remote_code=False,
                    local_files_only=config.get("offline", False))
            else:
                hf_config = dict(config["hf_config"])
                model_type = hf_config.pop("model_type")
                self.encoder = AutoModel.from_config(AutoConfig.for_model(model_type, **hf_config),
                                                     trust_remote_code=False)
            self.config["hf_config"] = self.encoder.config.to_dict()
            text_dim = self.encoder.config.hidden_size
        else:
            raise ValueError("backbone must be hashcnn or hf")
        hidden = config.get("hidden_dim", 128)
        self.head = nn.Sequential(nn.Linear(text_dim + config.get("dense_dim", 0), hidden),
                                  nn.ReLU(), nn.Dropout(config.get("dropout", 0.2)),
                                  nn.Linear(hidden, 1 if self.objective == "mse" else 5))
        if self.objective == "mse":
            nn.init.constant_(self.head[-1].bias, config.get("label_mean", 3.0))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                dense: torch.Tensor | None = None) -> torch.Tensor:
        if self.config["backbone"] == "hashcnn":
            text = self.encoder(input_ids, attention_mask)
        else:
            states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(states.dtype)
            text = (states * mask).sum(1) / mask.sum(1).clamp_min(1)
        if self.config.get("dense_dim", 0):
            if dense is None or dense.shape[-1] != self.config["dense_dim"]:
                raise ValueError("dense features do not match checkpoint architecture")
            text = torch.cat((text, dense.to(text.dtype)), dim=-1)
        return self.head(text)

    def loss(self, output: torch.Tensor, rating: torch.Tensor) -> torch.Tensor:
        rating = rating.float()
        if not torch.isfinite(rating).all() or (rating < 1).any() or (rating > 5).any():
            raise ValueError("training/evaluation requires finite ratings in [1,5]")
        if self.objective == "mse":
            return F.mse_loss(output.float().squeeze(-1), rating)
        if not torch.equal(rating, rating.round()):
            raise ValueError("classification targets must be integer ratings")
        target = rating.long() - 1
        if self.objective == "ce":
            return F.cross_entropy(output.float(), target)
        probabilities = observed_probabilities(output, self.transition)
        return F.nll_loss(probabilities.clamp_min(1e-12).log(), target)

    def predictions(self, output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Observed score is the metric target; latent score is diagnostic only."""
        if self.objective == "mse":
            observed = output.float().squeeze(-1).clamp(1, 5)
            return observed, torch.full_like(observed, float("nan"))
        probabilities = output.float().softmax(-1)
        if self.objective == "forward":
            return (probabilities @ self.transition.float()) @ self.stars, probabilities @ self.stars
        observed = probabilities @ self.stars
        return observed, torch.full_like(observed, float("nan"))
