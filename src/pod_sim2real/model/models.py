from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PODBasis:
    mean: np.ndarray
    modes: np.ndarray

    def transform(self, x):
        return (x - self.mean) @ self.modes.T

    def inverse(self, coefficients):
        return coefficients @ self.modes + self.mean

    def save(self, path):
        np.savez(path, mean=self.mean, modes=self.modes)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["mean"], z["modes"])


def _fit_pod(values, rank):
    mean = values.mean(axis=0)
    _, _, vectors = np.linalg.svd(values - mean, full_matrices=False)
    return PODBasis(mean.astype(np.float32), vectors[:min(rank, len(vectors))].astype(np.float32))


def fit_pod_bases(trajectories, rank=32, resolution=(32, 64), sample_stride=10):
    result = []
    for channel in range(2):
        samples = []
        for trajectory in trajectories:
            values = trajectory.fields(resolution)[:, channel].reshape(trajectory.u.shape[0], -1)
            samples.append(values[::sample_stride].astype(np.float32))
        result.append(_fit_pod(np.concatenate(samples, axis=0), rank))
    return tuple(result)


def fit_pod_bases_from_dataset(dataset, rank=32, max_samples=64):
    """Fit POD bases only from bounded windows from the training split."""
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    count = min(len(dataset), max_samples)
    samples = [[], []]
    for index in np.linspace(0, len(dataset) - 1, count, dtype=int):
        x, y, _ = dataset[int(index)]
        fields = torch.cat((x, y), dim=0).numpy()
        for channel in range(2):
            samples[channel].append(fields[:, channel].reshape(fields.shape[0], -1))
    return tuple(_fit_pod(np.concatenate(channel, axis=0).astype(np.float32, copy=False), rank) for channel in samples)


class SimpleUNet(nn.Module):
    """Standard 2D U-Net forecasting model across space with time mapped through channels."""
    def __init__(self, channels=2, width=32, input_steps=20, output_steps=20, out_channels=None):
        super().__init__()
        out_channels = channels if out_channels is None else out_channels
        self.channels, self.out_channels = channels, out_channels
        self.input_steps, self.output_steps = input_steps, output_steps
        in_dim = channels * input_steps
        out_dim = out_channels * output_steps
        groups = min(4, width)
        while width % groups:
            groups -= 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_dim, width, 3, padding=1),
            nn.GroupNorm(groups, width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
        )
        self.down = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
        )
        self.up = nn.ConvTranspose2d(width * 2, width, 4, stride=2, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(width * 2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, out_dim, 1),
        )

    def forward(self, x):
        batch, steps, ch, height, width = x.shape
        x_in = x.reshape(batch, steps * ch, height, width)
        encoded = self.enc1(x_in)
        decoded = self.up(self.mid(self.down(encoded)))
        if decoded.shape[-2:] != encoded.shape[-2:]:
            decoded = F.interpolate(decoded, size=encoded.shape[2:], mode="bilinear", align_corners=False)
        out = self.out(torch.cat((decoded, encoded), dim=1))
        return out.reshape(batch, self.output_steps, self.out_channels, height, width)


class SpectralConv2d(nn.Module):
    """FNO low-frequency integral layer supporting non-square grids."""
    def __init__(self, in_channels, out_channels, modes_h=8, modes_w=12):
        super().__init__()
        self.modes_h, self.modes_w = modes_h, modes_w
        shape = (in_channels, out_channels, modes_h, modes_w)
        scale = 1 / max(in_channels * out_channels, 1)
        self.positive_real = nn.Parameter(scale * torch.randn(*shape))
        self.positive_imag = nn.Parameter(scale * torch.randn(*shape))
        self.negative_real = nn.Parameter(scale * torch.randn(*shape))
        self.negative_imag = nn.Parameter(scale * torch.randn(*shape))

    def forward(self, x):
        batch, _, height, width = x.shape
        orig_dtype = x.dtype
        spectrum = torch.fft.rfft2(x.float(), norm="ortho")
        output = torch.zeros(batch, self.positive_real.shape[1], height, width // 2 + 1, dtype=torch.cfloat, device=x.device)
        mh, mw = min(self.modes_h, height), min(self.modes_w, width // 2 + 1)
        pos_w = torch.complex(self.positive_real[:, :, :mh, :mw], self.positive_imag[:, :, :mh, :mw])
        neg_w = torch.complex(self.negative_real[:, :, :mh, :mw], self.negative_imag[:, :, :mh, :mw])
        output[:, :, :mh, :mw] = torch.einsum("bihw,iohw->bohw", spectrum[:, :, :mh, :mw], pos_w)
        output[:, :, -mh:, :mw] = torch.einsum("bihw,iohw->bohw", spectrum[:, :, -mh:, :mw], neg_w)
        return torch.fft.irfft2(output, s=(height, width), norm="ortho").to(dtype=orig_dtype)


class FNO2d(nn.Module):
    def __init__(self, channels=2, width=32, layers=4, modes_h=8, modes_w=12, input_steps=20, output_steps=20):
        super().__init__()
        self.channels, self.input_steps, self.output_steps = channels, input_steps, output_steps
        self.lift = nn.Conv2d(channels * input_steps, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes_h, modes_w) for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, channels * output_steps, 1))

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        if (steps, channels) != (self.input_steps, self.channels):
            raise ValueError(f"expected [B, {self.input_steps}, {self.channels}, H, W]")
        z = self.lift(x.reshape(batch, steps * channels, height, width))
        for index, (spectral, pointwise) in enumerate(zip(self.spectral, self.pointwise)):
            z = spectral(z) + pointwise(z)
            if index + 1 < len(self.spectral):
                z = F.gelu(z)
        return self.project(z).reshape(batch, self.output_steps, self.channels, height, width)


class AdaptiveFourierFilter2d(nn.Module):
    """AFNO block-diagonal complex Fourier mixing from the supplied reference."""
    def __init__(self, channels, blocks=8, sparsity_threshold=0.01):
        super().__init__()
        if channels % blocks:
            raise ValueError("AFNO channels must be divisible by blocks")
        self.blocks, self.block_size = blocks, channels // blocks
        shape = (blocks, self.block_size, self.block_size, 2)
        self.w1 = nn.Parameter(0.02 * torch.randn(*shape))
        self.b1 = nn.Parameter(0.02 * torch.randn(blocks, self.block_size, 2))
        self.w2 = nn.Parameter(0.02 * torch.randn(*shape))
        self.b2 = nn.Parameter(0.02 * torch.randn(blocks, self.block_size, 2))
        self.sparsity_threshold = sparsity_threshold

    def forward(self, x):
        batch, height, width, channels = x.shape
        spectrum = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
        freq_width = spectrum.shape[2]
        values = torch.view_as_real(spectrum).reshape(batch, height, freq_width, self.blocks, self.block_size, 2)
        real, imag = values[..., 0], values[..., 1]
        first_real = torch.einsum("nhwgi,gio->nhwgo", real, self.w1[..., 0]) - torch.einsum("nhwgi,gio->nhwgo", imag, self.w1[..., 1]) + self.b1[..., 0]
        first_imag = torch.einsum("nhwgi,gio->nhwgo", real, self.w1[..., 1]) + torch.einsum("nhwgi,gio->nhwgo", imag, self.w1[..., 0]) + self.b1[..., 1]
        first_real, first_imag = F.gelu(first_real), F.gelu(first_imag)
        out_real = torch.einsum("nhwgi,gio->nhwgo", first_real, self.w2[..., 0]) - torch.einsum("nhwgi,gio->nhwgo", first_imag, self.w2[..., 1]) + self.b2[..., 0]
        out_imag = torch.einsum("nhwgi,gio->nhwgo", first_real, self.w2[..., 1]) + torch.einsum("nhwgi,gio->nhwgo", first_imag, self.w2[..., 0]) + self.b2[..., 1]
        output = F.softshrink(torch.stack((out_real, out_imag), dim=-1), lambd=self.sparsity_threshold)
        output = torch.view_as_complex(output.contiguous()).reshape(batch, height, freq_width, channels)
        return torch.fft.irfft2(output, s=(height, width), dim=(1, 2), norm="ortho").to(dtype=x.dtype)


class AFNOBlock(nn.Module):
    def __init__(self, channels, blocks, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.filter = AdaptiveFourierFilter2d(channels, blocks)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(channels * 2, channels), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.filter(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class AFNO2d(nn.Module):
    def __init__(self, channels=2, width=32, layers=4, blocks=8, input_steps=20, output_steps=20):
        super().__init__()
        self.channels, self.input_steps, self.output_steps = channels, input_steps, output_steps
        blocks = min(blocks, width)
        while width % blocks:
            blocks -= 1
        self.lift = nn.Conv2d(channels * input_steps, width, 1)
        self.layers = nn.ModuleList(AFNOBlock(width, blocks) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv2d(width, width * 2, 1), nn.GELU(), nn.Conv2d(width * 2, channels * output_steps, 1))

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        if (steps, channels) != (self.input_steps, self.channels):
            raise ValueError(f"expected [B, {self.input_steps}, {self.channels}, H, W]")
        z = self.lift(x.reshape(batch, steps * channels, height, width)).permute(0, 2, 3, 1)
        for layer in self.layers:
            z = layer(z)
        return self.project(z.permute(0, 3, 1, 2)).reshape(batch, self.output_steps, self.channels, height, width)


class PhysicsAttention(nn.Module):
    """Device-safe irregular Transolver attention: slice, attend, deslice."""
    def __init__(self, dim, heads=4, slice_num=16, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("Transolver dimension must be divisible by heads")
        self.heads, self.dim_head, self.scale = heads, dim // heads, (dim // heads) ** -0.5
        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_fx = nn.Linear(dim, dim)
        self.in_project_slice = nn.Linear(self.dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, _ = x.shape
        feature = self.in_project_fx(x).reshape(batch, tokens, self.heads, self.dim_head).transpose(1, 2)
        locations = self.in_project_x(x).reshape(batch, tokens, self.heads, self.dim_head).transpose(1, 2)
        weights = torch.softmax(self.in_project_slice(locations) / self.temperature.clamp(0.1, 5.0), dim=-1)
        slices = torch.einsum("bhnc,bhng->bhgc", feature, weights) / weights.sum(dim=2).clamp_min(1e-5).unsqueeze(-1)
        attention = torch.softmax((self.to_q(slices) @ self.to_k(slices).transpose(-1, -2)) * self.scale, dim=-1)
        output = torch.einsum("bhgc,bhng->bhnc", self.dropout(attention) @ self.to_v(slices), weights)
        return self.to_out(output.transpose(1, 2).reshape(batch, tokens, -1))


class TransolverBlock(nn.Module):
    def __init__(self, dim, heads, slice_num, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = PhysicsAttention(dim, heads, slice_num, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class InvertedTransolver(nn.Module):
    """Full-field inverted Transolver: each spatial/channel variable is a token."""
    def __init__(self, channels=2, width=32, input_steps=20, output_steps=20, depth=3, heads=4, dropout=0.0, slice_num=32):
        super().__init__()
        if width % heads:
            raise ValueError("attention width must be divisible by heads")
        self.channels, self.input_steps, self.output_steps = channels, input_steps, output_steps
        self.embed = nn.Linear(input_steps, width)
        self.channel_embedding = nn.Embedding(channels, width)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(TransolverBlock(width, heads, slice_num, dropout) for _ in range(depth))
        self.head = nn.Linear(width, output_steps)

    def forward(self, x):
        batch, steps, channels, height, grid_width = x.shape
        if (steps, channels) != (self.input_steps, self.channels):
            raise ValueError(f"expected [B, {self.input_steps}, {self.channels}, H, W]")
        variables = channels * height * grid_width
        tokens = x.permute(0, 2, 3, 4, 1).reshape(batch, variables, steps)
        means = tokens.mean(dim=-1, keepdim=True).detach()
        scales = tokens.var(dim=-1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
        tokens = (tokens - means) / scales
        channel_ids = torch.arange(channels, device=x.device).repeat_interleave(height * grid_width)
        tokens = self.dropout(self.embed(tokens) + self.channel_embedding(channel_ids).unsqueeze(0))
        for block in self.blocks:
            tokens = block(tokens)
        forecast = self.head(tokens) * scales + means
        return forecast.transpose(1, 2).reshape(batch, self.output_steps, channels, height, grid_width)


class SpectralConv1d(nn.Module):
    """1D Fourier spectral convolution layer for coefficient time series."""
    def __init__(self, in_channels, out_channels, modes=8):
        super().__init__()
        self.in_channels, self.out_channels, self.modes = in_channels, out_channels, modes
        scale = 1 / max(in_channels * out_channels, 1)
        self.weights_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weights_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x):
        # x: [B, in_channels, T]
        batch, _, length = x.shape
        orig_dtype = x.dtype
        spectrum = torch.fft.rfft(x.float(), norm="ortho")
        modes = min(self.modes, spectrum.shape[-1])
        weights = torch.complex(self.weights_real[:, :, :modes], self.weights_imag[:, :, :modes])
        out_spectrum = torch.zeros(batch, self.out_channels, spectrum.shape[-1], dtype=torch.cfloat, device=x.device)
        out_spectrum[:, :, :modes] = torch.einsum("bit,iot->bot", spectrum[:, :, :modes], weights)
        return torch.fft.irfft(out_spectrum, n=length, norm="ortho").to(dtype=orig_dtype)


class FNO1dCoeff(nn.Module):
    """1D FNO operator across POD modal coefficients and time with full modal mixing."""
    def __init__(self, rank, width=32, layers=4, modes=8, input_steps=20, output_steps=20):
        super().__init__()
        self.variables = 2 * rank
        self.input_steps, self.output_steps = input_steps, output_steps
        self.lift = nn.Linear(self.variables, width)
        self.spectral = nn.ModuleList(SpectralConv1d(width, width, modes) for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
        self.time_project = nn.Linear(input_steps, output_steps) if input_steps != output_steps else nn.Identity()
        self.project = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, self.variables)
        )

    def forward(self, x):
        # x: [B, input_steps, 2*rank]
        z = self.lift(x).transpose(1, 2)  # [B, width, input_steps]
        for i, (spectral, pointwise) in enumerate(zip(self.spectral, self.pointwise)):
            z = spectral(z) + pointwise(z)
            if i + 1 < len(self.spectral):
                z = F.gelu(z)
        z = self.time_project(z).transpose(1, 2)  # [B, output_steps, width]
        return self.project(z)  # [B, output_steps, 2*rank]


class UNet1dCoeff(nn.Module):
    """1D U-Net operator across POD modal coefficients and time.

    Operates strictly on temporal modal coefficients [B, input_steps, 2*rank] -> [B, output_steps, 2*rank].
    Does not participate in spatial reconstruction, matching POD-FNO / POD-AFNO design.
    """
    def __init__(self, rank, width=32, input_steps=20, output_steps=20):
        super().__init__()
        self.variables = 2 * rank
        self.input_steps, self.output_steps = input_steps, output_steps
        self.lift = nn.Linear(self.variables, width)

        groups = min(4, width)
        while width % groups:
            groups -= 1

        self.enc1 = nn.Sequential(
            nn.Conv1d(width, width * 2, 3, padding=1),
            nn.GroupNorm(groups, width * 2),
            nn.GELU(),
            nn.Conv1d(width * 2, width * 2, 3, padding=1),
            nn.GELU(),
        )
        self.down = nn.MaxPool1d(2)
        self.mid = nn.Sequential(
            nn.Conv1d(width * 2, width * 4, 3, padding=1),
            nn.GroupNorm(groups, width * 4),
            nn.GELU(),
            nn.Conv1d(width * 4, width * 4, 3, padding=1),
            nn.GELU(),
        )
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv1d(width * 4 + width * 2, width * 2, 3, padding=1),
            nn.GroupNorm(groups, width * 2),
            nn.GELU(),
            nn.Conv1d(width * 2, width, 3, padding=1),
            nn.GELU(),
        )
        self.time_project = nn.Linear(input_steps, output_steps) if input_steps != output_steps else nn.Identity()
        self.project = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, self.variables),
        )

    def forward(self, x):
        # x: [B, input_steps, variables]
        z = self.lift(x).transpose(1, 2)  # [B, width, input_steps]
        e1 = self.enc1(z)                 # [B, width * 2, input_steps]
        d1 = self.down(e1)                # [B, width * 2, input_steps // 2]
        m = self.mid(d1)                  # [B, width * 4, input_steps // 2]
        u1 = self.up(m)                   # [B, width * 4, input_steps]
        if u1.shape[-1] != e1.shape[-1]:
            u1 = F.interpolate(u1, size=e1.shape[-1], mode="linear", align_corners=False)
        cat = torch.cat([u1, e1], dim=1)
        out = self.dec1(cat)              # [B, width, input_steps]
        out = self.time_project(out).transpose(1, 2)  # [B, output_steps, width]
        return self.project(out)                      # [B, output_steps, variables]


class PODModel(nn.Module):
    def __init__(self, bases, kind="fno", width=32, input_steps=20, output_steps=20, depth=3, heads=4, dropout=0.0, slice_num=16, modes=8):
        super().__init__()
        if len(bases) != 2 or bases[0].modes.shape[0] != bases[1].modes.shape[0]:
            raise ValueError("POD models require equally ranked u/v bases")
        self.kind, self.rank = kind, bases[0].modes.shape[0]
        self.input_steps, self.output_steps = input_steps, output_steps
        self.register_buffer("mean", torch.from_numpy(np.stack([basis.mean for basis in bases]).astype(np.float32)))
        self.register_buffer("modes", torch.from_numpy(np.stack([basis.modes for basis in bases]).astype(np.float32)))
        variables = 2 * self.rank
        if kind == "fno":
            self.net = FNO1dCoeff(self.rank, width=width, layers=depth, modes=modes, input_steps=input_steps, output_steps=output_steps)
        elif kind == "afno":
            blocks = min(8, width)
            while width % blocks:
                blocks -= 1
            self.embed, self.net, self.head = nn.Linear(input_steps, width), nn.ModuleList(AFNOBlock(width, blocks, dropout) for _ in range(depth)), nn.Linear(width, output_steps)
        elif kind == "unet":
            self.net = UNet1dCoeff(self.rank, width=width, input_steps=input_steps, output_steps=output_steps)
        elif kind in {"itransformer", "itransolver"}:
            if width % heads:
                raise ValueError("attention width must be divisible by heads")
            self.embed = nn.Linear(input_steps, width)
            self.channel_embedding, self.mode_embedding = nn.Embedding(2, width), nn.Embedding(self.rank, width)
            self.dropout, self.head = nn.Dropout(dropout), nn.Linear(width, output_steps)
            if kind == "itransformer":
                layer = nn.TransformerEncoderLayer(width, heads, width * 2, dropout, batch_first=True, norm_first=True, activation="gelu")
                self.net = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            else:
                self.net = nn.ModuleList(TransolverBlock(width, heads, min(slice_num, variables), dropout) for _ in range(depth))
        else:
            raise ValueError(f"unknown POD kind {kind}")

    def coeff(self, x):
        batch, steps, channels, _, _ = x.shape
        if channels != 2:
            raise ValueError("POD models require u/v fields")
        flat = x.reshape(batch, steps, channels, -1)
        if flat.shape[-1] != self.mean.shape[-1]:
            raise ValueError("field resolution does not match POD basis")
        return torch.einsum("btcp,crp->btcr", flat - self.mean[None, None], self.modes).reshape(batch, steps, -1)

    def field(self, coefficients, height, width):
        batch, steps, variables = coefficients.shape
        if variables != 2 * self.rank:
            raise ValueError("coefficient dimension does not match POD rank")
        flat = torch.einsum("btcr,crp->btcp", coefficients.reshape(batch, steps, 2, self.rank), self.modes) + self.mean[None, None]
        return flat.reshape(batch, steps, 2, height, width)

    def _identity_embeddings(self, device):
        channels = torch.arange(2, device=device).repeat_interleave(self.rank)
        modes = torch.arange(self.rank, device=device).repeat(2)
        return self.channel_embedding(channels) + self.mode_embedding(modes)

    def _forecast_coefficients(self, coefficients):
        _, steps, variables = coefficients.shape
        if (steps, variables) != (self.input_steps, 2 * self.rank):
            raise ValueError("coefficient sequence does not match model configuration")
        if self.kind in {"fno", "unet"}:
            return self.net(coefficients)
        tokens = coefficients.transpose(1, 2)
        if self.kind == "afno":
            tokens = self.embed(tokens).unsqueeze(1)
            for layer in self.net:
                tokens = layer(tokens)
            return self.head(tokens.squeeze(1)).transpose(1, 2)
        means = tokens.mean(dim=-1, keepdim=True).detach()
        scales = tokens.var(dim=-1, keepdim=True, unbiased=False).add(1e-5).sqrt().detach()
        tokens = self.dropout(self.embed((tokens - means) / scales) + self._identity_embeddings(coefficients.device).unsqueeze(0))
        if self.kind == "itransformer":
            tokens = self.net(tokens)
        else:
            for layer in self.net:
                tokens = layer(tokens)
        return (self.head(tokens) * scales + means).transpose(1, 2)

    def forward(self, x):
        _, steps, _, height, width = x.shape
        if steps != self.input_steps:
            raise ValueError(f"expected {self.input_steps} input steps")
        coefficients = self.coeff(x)
        return self.field(self._forecast_coefficients(coefficients), height, width)


class ContinuousNeuralFieldDecoder(nn.Module):
    """Continuous coordinate-based neural field decoder for high-frequency spatial residuals."""
    def __init__(self, latent_dim, hidden_dim=64, out_channels=2):
        super().__init__()
        self.coord_proj = nn.Linear(2, hidden_dim)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_channels),
        )

    def forward(self, z_latent, height, width, device):
        # z_latent: [B, T_out, latent_dim]
        grid_y = torch.linspace(-1.0, 1.0, height, device=device, dtype=z_latent.dtype)
        grid_x = torch.linspace(-1.0, 1.0, width, device=device, dtype=z_latent.dtype)
        coords = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=-1)  # [H, W, 2]

        coord_feat = self.coord_proj(coords)  # [H, W, hidden_dim]
        latent_feat = self.latent_proj(z_latent)  # [B, T, hidden_dim]

        fused = latent_feat[:, :, None, None, :] + coord_feat[None, None, :, :, :]
        out = self.mlp(fused)  # [B, T, H, W, 2]
        return out.permute(0, 1, 4, 2, 3)  # [B, T, 2, H, W]


class TriadMNO(nn.Module):
    """Triad-Coupled Modal Neural Operator with Continuous Spatial Residual Fields.

    Triple Decomposition:
      u(x, y, t) = U_mean(x, y) + u_macro(x, y, t) + Delta_u(x, y, t)
    """
    def __init__(
        self,
        bases,
        width=32,
        input_steps=20,
        output_steps=20,
        macro_rank=None,
        micro_rank=None,
        depth=3,
        heads=4,
        dropout=0.0,
        use_triad_attn=True,
        use_continuous_field=True,
        **kwargs,
    ):
        super().__init__()
        if len(bases) != 2:
            raise ValueError("TriadMNO requires 2 bases for u and v fields")
        total_rank = bases[0].modes.shape[0]
        if macro_rank is None:
            macro_rank = min(32, total_rank // 2) if total_rank > 8 else max(1, total_rank // 2)
        micro_rank = total_rank - macro_rank if micro_rank is None else micro_rank
        self.macro_rank = macro_rank
        self.micro_rank = micro_rank
        self.input_steps, self.output_steps = input_steps, output_steps
        self.use_triad_attn = bool(use_triad_attn)
        self.use_continuous_field = bool(use_continuous_field)

        self.register_buffer("mean", torch.from_numpy(np.stack([b.mean for b in bases]).astype(np.float32)))
        self.register_buffer("modes_macro", torch.from_numpy(np.stack([b.modes[:macro_rank] for b in bases]).astype(np.float32)))
        if micro_rank > 0:
            self.register_buffer("modes_micro", torch.from_numpy(np.stack([b.modes[macro_rank:macro_rank+micro_rank] for b in bases]).astype(np.float32)))
        else:
            self.register_buffer("modes_micro", torch.zeros(2, 0, self.modes_macro.shape[-1], dtype=torch.float32))

        self.macro_vars = 2 * macro_rank
        self.micro_vars = 2 * micro_rank

        self.macro_net = FNO1dCoeff(macro_rank, width=width, layers=depth, modes=min(8, input_steps // 2 + 1), input_steps=input_steps, output_steps=output_steps)

        if micro_rank > 0:
            self.micro_lift = nn.Linear(self.micro_vars, width)
            self.micro_time_proj = nn.Linear(input_steps, output_steps) if input_steps != output_steps else nn.Identity()

            self.macro_proj = nn.Linear(self.macro_vars, width)
            self.triad_macro_to_micro = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.triad_micro_to_macro = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            self.triad_norm1 = nn.LayerNorm(width)
            self.triad_norm2 = nn.LayerNorm(width)

            if self.use_continuous_field:
                self.neural_field = ContinuousNeuralFieldDecoder(latent_dim=width, hidden_dim=width * 2, out_channels=2)
            else:
                self.micro_linear_head = nn.Linear(width, self.micro_vars)
        else:
            self.neural_field = None

    def freeze_macro(self, freeze: bool = True):
        """Freeze macro temporal operator for Sim2Real domain adaptation (Strategy A)."""
        for p in self.macro_net.parameters():
            p.requires_grad = not freeze
        if hasattr(self, "macro_proj"):
            for p in self.macro_proj.parameters():
                p.requires_grad = not freeze

    def coeff_macro(self, x):
        batch, steps, channels, _, _ = x.shape
        flat = x.reshape(batch, steps, channels, -1)
        return torch.einsum("btcp,crp->btcr", flat - self.mean[None, None], self.modes_macro).reshape(batch, steps, -1)

    def coeff_micro(self, x):
        if self.micro_rank == 0:
            return None
        batch, steps, channels, _, _ = x.shape
        flat = x.reshape(batch, steps, channels, -1)
        return torch.einsum("btcp,crp->btcr", flat - self.mean[None, None], self.modes_micro).reshape(batch, steps, -1)

    def field_macro(self, coefficients, height, width):
        batch, steps, variables = coefficients.shape
        flat = torch.einsum("btcr,crp->btcp", coefficients.reshape(batch, steps, 2, self.macro_rank), self.modes_macro) + self.mean[None, None]
        return flat.reshape(batch, steps, 2, height, width)

    def field_micro(self, coefficients, height, width):
        batch, steps, variables = coefficients.shape
        flat = torch.einsum("btcr,crp->btcp", coefficients.reshape(batch, steps, 2, self.micro_rank), self.modes_micro)
        return flat.reshape(batch, steps, 2, height, width)

    def forward(self, x):
        batch, steps, _, height, width = x.shape
        if steps != self.input_steps:
            raise ValueError(f"expected {self.input_steps} input steps")

        c_macro = self.coeff_macro(x)
        pred_macro = self.macro_net(c_macro)
        u_macro = self.field_macro(pred_macro, height, width)

        if self.micro_rank == 0:
            return u_macro

        c_micro = self.coeff_micro(x)
        z_micro = self.micro_time_proj(self.micro_lift(c_micro).transpose(1, 2)).transpose(1, 2)
        z_macro = self.macro_proj(pred_macro)

        if self.use_triad_attn:
            attn_micro, _ = self.triad_macro_to_micro(z_micro, z_macro, z_macro)
            z_micro_fused = self.triad_norm1(z_micro + attn_micro)
        else:
            z_micro_fused = z_micro

        if self.use_continuous_field:
            delta_u = self.neural_field(z_micro_fused, height, width, x.device)
        else:
            c_micro_pred = self.micro_linear_head(z_micro_fused)
            delta_u = self.field_micro(c_micro_pred, height, width)

        return u_macro + delta_u


def build_model(name, bases=None, width=32, input_steps=20, output_steps=20, **options):
    name = name.lower()
    common = {"width": width, "input_steps": input_steps, "output_steps": output_steps}
    if name == "unet":
        return SimpleUNet(2, width, input_steps=input_steps, output_steps=output_steps)
    if name == "fno":
        return FNO2d(**common)
    if name == "afno":
        return AFNO2d(**common, layers=int(options.get("depth", 4)))
    if name == "itransolver":
        return InvertedTransolver(**common, depth=int(options.get("depth", 3)), heads=int(options.get("heads", 4)), dropout=float(options.get("dropout", 0.0)), slice_num=int(options.get("slice_num", 32)))
    if name in {"triad-mno", "triad_mno"}:
        if bases is None:
            raise ValueError("Triad-MNO requires fitted bases")
        return TriadMNO(
            bases,
            **common,
            depth=int(options.get("depth", 3)),
            heads=int(options.get("heads", 4)),
            dropout=float(options.get("dropout", 0.0)),
            macro_rank=options.get("macro_rank"),
            micro_rank=options.get("micro_rank"),
            use_triad_attn=options.get("use_triad_attn", True),
            use_continuous_field=options.get("use_continuous_field", True),
        )
    if name.startswith("pod-"):
        if bases is None:
            raise ValueError("POD models require fitted bases")
        kind = name.removeprefix("pod-")
        if kind == "transolver":
            kind = "itransolver"
        return PODModel(
            bases,
            kind=kind,
            **common,
            depth=int(options.get("depth", 3)),
            heads=int(options.get("heads", 4)),
            dropout=float(options.get("dropout", 0.0)),
            slice_num=int(options.get("slice_num", 16)),
            modes=int(options.get("modes", 8)),
        )
    raise ValueError(f"unknown model {name}")

