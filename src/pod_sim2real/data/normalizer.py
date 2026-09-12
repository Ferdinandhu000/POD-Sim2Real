from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np
import torch


def _broadcast_to(stat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast 1D stat tensor of shape (C,) to match the channel axis of target tensor."""
    stat = stat.to(device=target.device, dtype=target.dtype)
    c = stat.numel()

    # 5D tensor: [B, T, C, H, W] (default) or [B, T, H, W, C] (RealPDEBench style)
    if target.ndim == 5:
        if target.shape[2] == c:
            return stat.view(1, 1, c, 1, 1)
        elif target.shape[-1] == c:
            return stat.view(1, 1, 1, 1, c)
        else:
            raise ValueError(f"Cannot align stat of shape ({c},) with target shape {target.shape}")

    # 4D tensor: [T, C, H, W] or [B, C, H, W] or [T, H, W, C]
    elif target.ndim == 4:
        if target.shape[1] == c:
            return stat.view(1, c, 1, 1)
        elif target.shape[-1] == c:
            return stat.view(1, 1, 1, c)
        else:
            raise ValueError(f"Cannot align stat of shape ({c},) with target shape {target.shape}")

    # 3D tensor: [C, H, W] or [H, W, C]
    elif target.ndim == 3:
        if target.shape[0] == c:
            return stat.view(c, 1, 1)
        elif target.shape[-1] == c:
            return stat.view(1, 1, c)

    return stat


class IdentityNormalizer:
    """Pass-through identity normalizer for backward compatibility."""
    def __init__(self, device: torch.device | str | None = None):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.mean = None
        self.std = None

    def to(self, device: torch.device | str) -> IdentityNormalizer:
        self.device = torch.device(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def preprocess(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if y is None:
            return x
        return x, y

    def postprocess(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if y is None:
            return x
        return x, y

    def state_dict(self) -> dict:
        return {"normalizer_type": "identity"}


class GaussianNormalizer:
    """Channel-wise Gaussian normalizer (aligns with RealPDEBench data_normalizer).

    Computes or receives channel-wise mean and std:
        x_norm = (x - mean) / std
        x = x_norm * std + mean
    """
    def __init__(
        self,
        mean: torch.Tensor | Sequence[float] | np.ndarray,
        std: torch.Tensor | Sequence[float] | np.ndarray,
        device: torch.device | str | None = None,
        eps: float = 1e-8,
    ):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)

        mean = mean.view(-1).float()
        std = std.view(-1).float()
        std = torch.where(std == 0, torch.ones_like(std), std)
        std = torch.clamp(std, min=eps)

        self.mean = mean.to(self.device)
        self.std = std.to(self.device)
        self.eps = eps

    def to(self, device: torch.device | str) -> GaussianNormalizer:
        self.device = torch.device(device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean_b = _broadcast_to(self.mean, x)
        std_b = _broadcast_to(self.std, x)
        return (x - mean_b) / std_b

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean_b = _broadcast_to(self.mean, x)
        std_b = _broadcast_to(self.std, x)
        return x * std_b + mean_b

    def preprocess(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if y is None:
            return self.normalize(x)
        return self.normalize(x), self.normalize(y)

    def postprocess(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if y is None:
            return self.denormalize(x)
        return self.denormalize(x), self.denormalize(y)

    def state_dict(self) -> dict:
        return {
            "normalizer_type": "gaussian",
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
        }

    def save(self, path: Path | str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def from_state_dict(
        cls, state: dict, device: torch.device | str | None = None
    ) -> GaussianNormalizer:
        if "mean" not in state or "std" not in state:
            raise KeyError(f"Invalid state dict for GaussianNormalizer: keys found: {list(state.keys())}")
        return cls(state["mean"], state["std"], device=device)

    @classmethod
    def from_file(
        cls, path: Path | str, device: torch.device | str | None = None
    ) -> GaussianNormalizer:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_state_dict(payload, device=device)

    @classmethod
    def from_dataset(
        cls,
        dataset,
        device: torch.device | str | None = None,
        max_samples: int = 128,
        num_channels: int = 2,
    ) -> GaussianNormalizer:
        """Compute channel-wise mean and std from dataset windows."""
        count = min(len(dataset), max_samples)
        if count <= 0:
            raise ValueError("Dataset is empty")
        indices = np.linspace(0, len(dataset) - 1, count, dtype=int)

        totals = np.zeros(num_channels, dtype=np.float64)
        totals_sq = np.zeros(num_channels, dtype=np.float64)
        n_elements = np.zeros(num_channels, dtype=np.int64)

        for idx in indices:
            sample = dataset[int(idx)]
            # dataset item is (x, y, meta) or (x, y)
            x, y = sample[0], sample[1]
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(x, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y, dtype=torch.float32)

            # Combine input and target frames along time dimension
            frames = torch.cat([x, y], dim=0)

            if frames.ndim == 4 and frames.shape[1] == num_channels:
                # [T, C, H, W]
                for c in range(num_channels):
                    vals = frames[:, c].reshape(-1).double().numpy()
                    totals[c] += vals.sum()
                    totals_sq[c] += np.square(vals).sum()
                    n_elements[c] += vals.size
            elif frames.ndim == 4 and frames.shape[-1] == num_channels:
                # [T, H, W, C]
                for c in range(num_channels):
                    vals = frames[..., c].reshape(-1).double().numpy()
                    totals[c] += vals.sum()
                    totals_sq[c] += np.square(vals).sum()
                    n_elements[c] += vals.size
            else:
                raise ValueError(f"Unsupported tensor shape for normalizer: {frames.shape}")

        means = totals / np.maximum(n_elements, 1)
        vars_ = np.maximum((totals_sq / np.maximum(n_elements, 1)) - np.square(means), 1e-12)
        stds = np.sqrt(vars_)

        return cls(torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32), device=device)


def build_normalizer(
    normalizer_type: str = "none",
    dataset=None,
    checkpoint_or_path: Path | str | None = None,
    device: torch.device | str | None = None,
    max_samples: int = 128,
) -> IdentityNormalizer | GaussianNormalizer:
    """Build normalizer instance based on config and available statistics."""
    norm_type = (normalizer_type or "none").lower().strip()
    if norm_type in ("none", "identity"):
        return IdentityNormalizer(device=device)
    elif norm_type == "gaussian":
        if checkpoint_or_path is not None and Path(checkpoint_or_path).exists():
            try:
                payload = torch.load(checkpoint_or_path, map_location="cpu", weights_only=False)
                if isinstance(payload, dict) and "mean" in payload and "std" in payload:
                    return GaussianNormalizer.from_state_dict(payload, device=device)
            except Exception:
                pass
        if dataset is not None:
            return GaussianNormalizer.from_dataset(dataset, device=device, max_samples=max_samples)
        raise ValueError("GaussianNormalizer requires either a dataset or a valid checkpoint path containing mean/std")
    else:
        raise ValueError(f"Unsupported normalizer type: {normalizer_type}")
