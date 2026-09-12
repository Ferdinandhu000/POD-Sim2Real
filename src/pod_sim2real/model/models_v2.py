import math
from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import AFNOBlock, PODBasis


# ==============================================================================
# 1. Triad-AFNO: Dual-Tower AFNO with Cross-Scale Triad Attention
# ==============================================================================

class TriadAFNO(nn.Module):
    """Triad Adaptive Fourier Neural Operator (Triad-AFNO).

    Integrates AFNO's frequency-domain block-diagonal mixing and soft-shrinkage
    sparsity into the multi-scale Triad decomposition:
      - Macro-AFNO tower: captures large-scale coherent structure dynamics.
      - Micro-AFNO tower: filters high-frequency measurement noise via soft-thresholding.
      - Triad Mode-Cross-Attention: models Navier-Stokes energy cascade from Macro to Micro.
      - Supports both 'grid2d' ([B, 2, rank, width]) and 'flat1d' ([B, 1, 2*rank, width]) mode layouts.
    """
    def __init__(
        self,
        bases: Sequence[PODBasis],
        width: int = 32,
        input_steps: int = 20,
        output_steps: int = 20,
        depth: int = 3,
        heads: int = 4,
        macro_rank: int = 32,
        micro_rank: int = 32,
        mode_layout: str = "grid2d",  # 'grid2d' or 'flat1d'
        dropout: float = 0.0,
        sparsity_threshold: float = 0.01,
        **kwargs,
    ):
        super().__init__()
        if len(bases) != 2:
            raise ValueError("TriadAFNO requires 2 bases for u and v fields")
        total_rank = bases[0].modes.shape[0]
        self.macro_rank = min(macro_rank, total_rank)
        self.micro_rank = min(micro_rank, total_rank - self.macro_rank) if micro_rank is not None else (total_rank - self.macro_rank)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.mode_layout = mode_layout.lower()
        self.width = width
        self.depth = depth

        # Register POD spatial bases
        self.register_buffer("mean", torch.from_numpy(np.stack([b.mean for b in bases]).astype(np.float32)))
        self.register_buffer("modes_macro", torch.from_numpy(np.stack([b.modes[:self.macro_rank] for b in bases]).astype(np.float32)))
        if self.micro_rank > 0:
            self.register_buffer(
                "modes_micro",
                torch.from_numpy(np.stack([b.modes[self.macro_rank:self.macro_rank+self.micro_rank] for b in bases]).astype(np.float32))
            )
        else:
            self.register_buffer("modes_micro", torch.zeros(2, 0, self.modes_macro.shape[-1], dtype=torch.float32))

        blocks = min(8, width)
        while width % blocks:
            blocks -= 1

        # Temporal Lift & Projection
        self.embed_macro = nn.Linear(input_steps, width)
        self.head_macro = nn.Linear(width, output_steps)

        # Macro-AFNO Stack
        self.macro_net = nn.ModuleList([
            AFNOBlock(width, blocks, dropout=dropout) for _ in range(depth)
        ])

        if self.micro_rank > 0:
            self.embed_micro = nn.Linear(input_steps, width)
            self.head_micro = nn.Linear(width, output_steps)

            # Micro-AFNO Stack
            self.micro_net = nn.ModuleList([
                AFNOBlock(width, blocks, dropout=dropout) for _ in range(depth)
            ])

            # Triad Cross-Attention: Micro attends to Macro
            actual_heads = heads if (width % heads == 0) else max(1, math.gcd(width, heads))
            self.triad_cross_attn = nn.MultiheadAttention(width, actual_heads, dropout=dropout, batch_first=True)
            self.norm_triad = nn.LayerNorm(width)
            self.mlp_triad = nn.Sequential(
                nn.Linear(width, width * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width * 2, width),
                nn.Dropout(dropout),
            )
            self.norm_triad2 = nn.LayerNorm(width)

    def _coeff(self, x, modes):
        # x: [B, T, 2, H, W] -> flat: [B, T, 2, H*W]
        batch, steps, channels, _, _ = x.shape
        flat = x.reshape(batch, steps, channels, -1)
        # modes: [2, rank, H*W]
        return torch.einsum("btcp,crp->btcr", flat - self.mean[None, None], modes)

    def _field(self, c, modes):
        # c: [B, T, 2, rank], modes: [2, rank, H*W]
        return torch.einsum("btcr,crp->btcp", c, modes)

    def forward(self, x):
        batch, steps, channels, height, width = x.shape
        if steps != self.input_steps:
            raise ValueError(f"expected {self.input_steps} input steps, got {steps}")

        # 1. Project to Macro & Micro coefficients
        c_macro = self._coeff(x, self.modes_macro)  # [B, T, 2, macro_rank]

        # 2. Tokenization & Layout shaping
        # c_macro: [B, T, 2, M_macro] -> permute -> [B, 2, M_macro, T]
        t_macro = c_macro.permute(0, 2, 3, 1)  # [B, 2, M_macro, T_in]
        z_macro = self.embed_macro(t_macro)     # [B, 2, M_macro, width]

        if self.mode_layout == "grid2d":
            # Natural 2D grid: H=2 (u,v channels), W=macro_rank, C=width
            grid_macro = z_macro
        else:
            # 1D flat: H=1, W=2*macro_rank, C=width
            grid_macro = z_macro.reshape(batch, 1, 2 * self.macro_rank, self.width)

        # 3. Macro-AFNO Evolution
        for layer in self.macro_net:
            grid_macro = layer(grid_macro)

        # 4. Micro Processing & Triad Coupling
        if self.micro_rank > 0:
            c_micro = self._coeff(x, self.modes_micro)  # [B, T, 2, micro_rank]
            t_micro = c_micro.permute(0, 2, 3, 1)       # [B, 2, M_micro, T_in]
            z_micro = self.embed_micro(t_micro)          # [B, 2, M_micro, width]

            if self.mode_layout == "grid2d":
                grid_micro = z_micro
            else:
                grid_micro = z_micro.reshape(batch, 1, 2 * self.micro_rank, self.width)

            # Micro-AFNO Evolution (Intrinsic dynamics & soft-threshold denoising)
            for layer in self.micro_net:
                grid_micro = layer(grid_micro)

            # 5. Triad Cross-Scale Attention
            # Flatten tokens to [B, N_tokens, width]
            tokens_macro = grid_macro.reshape(batch, -1, self.width)
            tokens_micro = grid_micro.reshape(batch, -1, self.width)

            # Micro queries Macro for scale energy cascade
            attn_out, _ = self.triad_cross_attn(
                query=tokens_micro,
                key=tokens_macro,
                value=tokens_macro,
            )
            tokens_micro_fused = self.norm_triad(tokens_micro + attn_out)
            tokens_micro_fused = self.norm_triad2(tokens_micro_fused + self.mlp_triad(tokens_micro_fused))

            # Reshape back to micro layout
            if self.mode_layout == "grid2d":
                z_micro_final = tokens_micro_fused.reshape(batch, 2, self.micro_rank, self.width)
            else:
                z_micro_final = tokens_micro_fused.reshape(batch, 2, self.micro_rank, self.width)

            pred_c_micro = self.head_micro(z_micro_final).permute(0, 3, 1, 2)  # [B, T_out, 2, micro_rank]
            field_micro = self._field(pred_c_micro, self.modes_micro)
        else:
            field_micro = 0.0

        # Reshape macro back to [B, 2, macro_rank, width]
        z_macro_final = grid_macro.reshape(batch, 2, self.macro_rank, self.width)
        pred_c_macro = self.head_macro(z_macro_final).permute(0, 3, 1, 2)  # [B, T_out, 2, macro_rank]
        field_macro = self._field(pred_c_macro, self.modes_macro)

        # Final field reconstruction: Mean + Macro + Micro
        flat_total = self.mean[None, None] + field_macro + field_micro
        return flat_total.reshape(batch, self.output_steps, 2, height, width)


# ==============================================================================
# 2. 3D Neural Operator Baselines (RealPDEBench 100% Identical Implementations)
# ==============================================================================
from .realpdebench.fno import FNO3d as RealPDEBench_FNO3d
from .realpdebench.unet import Unet3d as RealPDEBench_Unet3d
from .realpdebench.afno import AFNO3d as RealPDEBench_AFNO3d


class FNO3d(nn.Module):
    """Exact RealPDEBench FNO3d baseline operator ensuring 100% architectural and mathematical alignment."""
    def __init__(
        self,
        channels: int = 2,
        width: int = 64,
        layers: int = 4,
        modes_t: int = 4,
        modes_h: int = 12,
        modes_w: int = 16,
        input_steps: int = 20,
        output_steps: int = 20,
        resolution: tuple[int, int] = (64, 128),
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.input_steps = input_steps
        self.output_steps = output_steps
        shape_in = (input_steps, resolution[0], resolution[1], channels)
        shape_out = (output_steps, resolution[0], resolution[1], channels)
        self.core = RealPDEBench_FNO3d(
            modes1=modes_t,
            modes2=modes_h,
            modes3=modes_w,
            n_layers=layers,
            width=width,
            shape_in=shape_in,
            shape_out=shape_out,
        )

    def forward(self, x):
        # x: [B, T_in, C, H, W] -> RealPDEBench natively expects [B, T_in, H, W, C]
        if x.shape[2] == self.channels:
            x_in = x.permute(0, 1, 3, 4, 2)
            out = self.core(x_in)
            return out.permute(0, 1, 4, 2, 3)
        return self.core(x)


class Unet3d(nn.Module):
    """Exact RealPDEBench Unet3d baseline operator with spatial-temporal attention and rotary embeddings."""
    def __init__(
        self,
        channels: int = 2,
        width: int = 64,
        input_steps: int = 20,
        output_steps: int = 20,
        dim_mults: tuple[int, ...] = (1, 2, 4),
        resolution: tuple[int, int] = (64, 128),
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.input_steps = input_steps
        self.output_steps = output_steps
        dim = int(kwargs.get("dim", min(resolution[0], width)))
        self.core = RealPDEBench_Unet3d(
            dim=dim,
            out_channels=channels,
            dim_mults=dim_mults,
            channels=channels,
            in_time=input_steps,
            out_time=output_steps,
        )

    def forward(self, x):
        # x: [B, T_in, C, H, W] -> RealPDEBench expects [B, T_in, H, W, C]
        if x.shape[2] == self.channels:
            x_in = x.permute(0, 1, 3, 4, 2)
            out = self.core(x_in)
            return out.permute(0, 1, 4, 2, 3)
        return self.core(x)


class AFNO3d(nn.Module):
    """3D AFNO operator strictly adhering to RealPDEBench FNO3d structure with Adaptive Fourier Blocks."""
    def __init__(
        self,
        channels: int = 2,
        width: int = 64,
        layers: int = 4,
        blocks: int = 8,
        sparsity_threshold: float = 0.01,
        input_steps: int = 20,
        output_steps: int = 20,
        resolution: tuple[int, int] = (64, 128),
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.input_steps = input_steps
        self.output_steps = output_steps
        shape_in = (input_steps, resolution[0], resolution[1], channels)
        shape_out = (output_steps, resolution[0], resolution[1], channels)
        self.core = RealPDEBench_AFNO3d(
            n_layers=layers,
            width=width,
            shape_in=shape_in,
            shape_out=shape_out,
            blocks=blocks,
            sparsity_threshold=sparsity_threshold,
        )

    def forward(self, x):
        # x: [B, T_in, C, H, W] -> RealPDEBench expects [B, T_in, H, W, C]
        if x.shape[2] == self.channels:
            x_in = x.permute(0, 1, 3, 4, 2)
            out = self.core(x_in)
            return out.permute(0, 1, 4, 2, 3)
        return self.core(x)

