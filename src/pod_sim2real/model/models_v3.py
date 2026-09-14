import math
from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import AFNOBlock, PODBasis
from .realpdebench.transformer import Transformer3d as RealPDEBench_Transformer3d
from .realpdebench.unet import Unet3d as RealPDEBench_Unet3d


class PODResTransformer3d(nn.Module):
    """POD-Driven Residual Spatiotemporal Operator (POD-ResTransformer3D).
    
    Integrates a full-rank POD-AFNO2D macro backbone with the exact RealPDEBench
    Transformer3d residual operator via dual-stream spatiotemporal concatenation.

    Key Architectural Principles:
      1. Complete POD Backbone (No Macro/Micro Split):
         - Single joint SVD spatial basis across u and v fields (Rank K=64 or 96).
         - AFNO-2D mode evolution network on grid2d [B, 2, rank, pod_width].
         - Direct linear modal field reconstruction -> u_pod: [B, T_out, 2, H, W].
      2. Exact Transformer3D Residual Module:
         - Uses RealPDEBench Transformer3d directly with SpaceTimeAttention,
           coordinate injection (t, x, y), and patch_size=(4, 4).
         - Dual-stream temporal concatenation: [u_in (past), u_pod (future)]
           forming an end-to-end 40-frame continuous flow cube -> delta_u.
      3. Additive Physics Field:
         - u_final = u_pod + delta_u.
    """
    def __init__(
        self,
        bases: Sequence[PODBasis],
        pod_rank: int = 96,
        pod_width: int = 32,
        pod_depth: int = 4,
        pod_dropout: float = 0.0,
        # Exact Transformer3D parameters from v4
        res_layers: int = 4,
        res_width: int = 64,
        res_heads: int = 8,
        res_patch_size: Tuple[int, int] = (4, 4),
        res_dropout: float = 0.0,
        input_steps: int = 20,
        output_steps: int = 20,
        resolution: Tuple[int, int] = (64, 128),
        return_aux: bool = False,
        dual_stream_mode: str = "temporal_concat",
        **kwargs,
    ):
        super().__init__()
        if len(bases) != 2:
            raise ValueError("PODResTransformer3d requires 2 bases for u and v fields")
        total_rank = bases[0].modes.shape[0]
        self.pod_rank = min(pod_rank, total_rank)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.resolution = resolution
        self.return_aux = return_aux
        self.dual_stream_mode = dual_stream_mode.lower()

        # Register POD spatial bases
        self.register_buffer("mean", torch.from_numpy(np.stack([b.mean for b in bases]).astype(np.float32)))
        self.register_buffer("modes", torch.from_numpy(np.stack([b.modes[:self.pod_rank] for b in bases]).astype(np.float32)))

        # 1. POD Backbone: AFNO-2D Mode Operator (Grid2D: [B, 2, rank, pod_width])
        blocks = min(8, pod_width)
        while pod_width % blocks:
            blocks -= 1
        self.embed_pod = nn.Linear(input_steps, pod_width)
        self.head_pod = nn.Linear(pod_width, output_steps)
        self.afno_layers = nn.ModuleList([
            AFNOBlock(pod_width, blocks=blocks, dropout=pod_dropout) for _ in range(pod_depth)
        ])

        # 2. Residual Module: Exact RealPDEBench Transformer3D
        h, w = resolution
        if self.dual_stream_mode == "temporal_concat":
            shape_in = (input_steps + output_steps, h, w, 2)
        elif self.dual_stream_mode == "channel_concat":
            shape_in = (input_steps, h, w, 4)
        else:
            shape_in = (input_steps + output_steps, h, w, 2)
        shape_out = (output_steps, h, w, 2)

        self.residual_transformer = RealPDEBench_Transformer3d(
            n_layers=res_layers,
            width=res_width,
            heads=res_heads,
            patch_size=res_patch_size,
            dropout=res_dropout,
            shape_in=shape_in,
            shape_out=shape_out,
        )

    def freeze_pod(self, freeze: bool = True):
        """Freeze POD backbone parameters."""
        for p in self.embed_pod.parameters():
            p.requires_grad = not freeze
        for p in self.head_pod.parameters():
            p.requires_grad = not freeze
        for p in self.afno_layers.parameters():
            p.requires_grad = not freeze

    def freeze_macro(self, freeze: bool = True):
        """Alias for train.py Strategy A compatibility."""
        self.freeze_pod(freeze)

    def freeze_residual(self, freeze: bool = True):
        """Freeze residual transformer parameters."""
        for p in self.residual_transformer.parameters():
            p.requires_grad = not freeze

    def _coeff(self, x, modes):
        batch, steps, channels, _, _ = x.shape
        flat = x.reshape(batch, steps, channels, -1)
        return torch.einsum("btcp,crp->btcr", flat - self.mean[None, None], modes)

    def _field(self, c, modes):
        return torch.einsum("btcr,crp->btcp", c, modes)

    def forward_pod(self, x):
        batch, steps, channels, h, w = x.shape
        c = self._coeff(x, self.modes)  # [B, T_in, 2, rank]
        grid = self.embed_pod(c.permute(0, 2, 3, 1))  # [B, 2, rank, pod_width]
        for layer in self.afno_layers:
            grid = layer(grid)
        pred_c = self.head_pod(grid).permute(0, 3, 1, 2)  # [B, T_out, 2, rank]
        u_pod_flat = self.mean[None, None] + self._field(pred_c, self.modes)
        return u_pod_flat.reshape(batch, self.output_steps, 2, h, w)

    def forward(self, x, return_aux=None):
        batch, steps, channels, h, w = x.shape
        # 1. Coarse POD field prediction
        u_pod = self.forward_pod(x)  # [B, T_out, 2, H, W]

        # 2. Dual-stream spatiotemporal assembly
        if self.dual_stream_mode == "temporal_concat":
            x_hwc = x.permute(0, 1, 3, 4, 2)          # [B, T_in, H, W, 2]
            u_pod_hwc = u_pod.permute(0, 1, 3, 4, 2)  # [B, T_out, H, W, 2]
            dual_in = torch.cat([x_hwc, u_pod_hwc], dim=1)  # [B, T_in + T_out, H, W, 2]
        elif self.dual_stream_mode == "channel_concat":
            x_hwc = x.permute(0, 1, 3, 4, 2)
            u_pod_hwc = u_pod.permute(0, 1, 3, 4, 2)
            dual_in = torch.cat([x_hwc, u_pod_hwc], dim=-1)  # [B, T, H, W, 4]
        else:
            x_hwc = x.permute(0, 1, 3, 4, 2)
            u_pod_hwc = u_pod.permute(0, 1, 3, 4, 2)
            dual_in = torch.cat([x_hwc, u_pod_hwc], dim=1)

        # 3. Residual learning via exact RealPDEBench Transformer3D
        delta_u_hwc = self.residual_transformer(dual_in)  # [B, T_out, H, W, 2]
        delta_u = delta_u_hwc.permute(0, 1, 4, 2, 3)     # [B, T_out, 2, H, W]

        # 4. Additive synthesis
        u_final = u_pod + delta_u

        ret_aux = return_aux if return_aux is not None else self.return_aux
        if ret_aux:
            return u_final, u_pod
        return u_final


class PODResUNet3d(nn.Module):
    """POD coarse prediction with a RealPDEBench 3D-UNet residual core.

    Dual-stream Spatiotemporal Coupling modes:
    1. 'temporal_concat' (default, matches v5 PODResTransformer3d):
       Concatenates history [B, T_in, H, W, 2] and coarse POD future [B, T_out, H, W, 2]
       along the temporal axis into a continuous 40-step fluid volume [B, 40, H, W, 2].
       The 3D-UNet processes this continuous physical flow and projects (40 -> 20)
       to produce high-frequency residual correction delta_u.
    2. 'future_refine':
       The 3D-UNet directly refines the coarse future field u_pod [B, T_out, H, W, 2] -> delta_u.
    3. 'channel_concat' (legacy):
       Concatenates history and coarse prediction along the channel dimension [B, T, H, W, 4].
    """

    def __init__(
        self,
        bases: Sequence[PODBasis],
        pod_rank: int = 96,
        pod_width: int = 32,
        pod_depth: int = 4,
        pod_dropout: float = 0.0,
        res_width: int = 64,
        res_dim_mults: Tuple[int, ...] = (1, 2, 4),
        res_attn_heads: int = 4,
        res_attn_dim_head: int = 32,
        resnet_groups: int = 8,
        input_steps: int = 20,
        output_steps: int = 20,
        resolution: Tuple[int, int] = (64, 128),
        dual_stream_mode: str = "temporal_concat",
        residual_input: str | None = None,
        return_aux: bool = False,
        **kwargs,
    ):
        super().__init__()
        if len(bases) != 2:
            raise ValueError("PODResUNet3d requires 2 bases for u and v fields")

        # Map legacy residual_input to dual_stream_mode if provided
        mode = (residual_input or dual_stream_mode).lower().strip()
        if mode not in {"temporal_concat", "future_refine", "channel_concat"}:
            raise ValueError(
                f"unsupported dual_stream_mode={mode!r}, expected 'temporal_concat', 'future_refine', or 'channel_concat'"
            )
        self.dual_stream_mode = mode

        total_rank = bases[0].modes.shape[0]
        self.pod_rank = min(pod_rank, total_rank)
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.resolution = tuple(resolution)
        self.return_aux = return_aux

        self.register_buffer("mean", torch.from_numpy(np.stack([b.mean for b in bases]).astype(np.float32)))
        self.register_buffer(
            "modes", torch.from_numpy(np.stack([b.modes[:self.pod_rank] for b in bases]).astype(np.float32))
        )

        blocks = min(8, pod_width)
        while pod_width % blocks:
            blocks -= 1
        self.embed_pod = nn.Linear(input_steps, pod_width)
        self.head_pod = nn.Linear(pod_width, output_steps)
        self.afno_layers = nn.ModuleList([
            AFNOBlock(pod_width, blocks=blocks, dropout=pod_dropout) for _ in range(pod_depth)
        ])

        height, width = self.resolution
        if self.dual_stream_mode == "temporal_concat":
            unet_in_time = input_steps + output_steps
            unet_channels = 2
            self.time_project = nn.Linear(unet_in_time, output_steps)
        elif self.dual_stream_mode == "future_refine":
            unet_in_time = output_steps
            unet_channels = 2
            self.time_project = None
        else:  # channel_concat
            if input_steps != output_steps:
                raise ValueError("PODResUNet3d channel_concat requires input_steps == output_steps")
            unet_in_time = input_steps
            unet_channels = 4
            self.time_project = None

        self.residual_unet = RealPDEBench_Unet3d(
            dim=res_width,
            out_channels=2,
            dim_mults=tuple(res_dim_mults),
            channels=unet_channels,
            attn_heads=res_attn_heads,
            attn_dim_head=res_attn_dim_head,
            resnet_groups=resnet_groups,
            in_time=unet_in_time,
            out_time=unet_in_time,
        )

    def freeze_pod(self, freeze: bool = True):
        for module in (self.embed_pod, self.head_pod, self.afno_layers):
            for parameter in module.parameters():
                parameter.requires_grad = not freeze

    def freeze_macro(self, freeze: bool = True):
        self.freeze_pod(freeze)

    def freeze_residual(self, freeze: bool = True):
        for parameter in self.residual_unet.parameters():
            parameter.requires_grad = not freeze
        if self.time_project is not None:
            for parameter in self.time_project.parameters():
                parameter.requires_grad = not freeze

    def _coeff(self, x):
        batch, steps, channels, _, _ = x.shape
        flat = x.reshape(batch, steps, channels, -1)
        return torch.einsum("btcp,crp->btcr", flat - self.mean[None, None], self.modes)

    def _field(self, coefficients):
        return torch.einsum("btcr,crp->btcp", coefficients, self.modes)

    def forward_pod(self, x):
        batch, _, _, height, width = x.shape
        coefficients = self._coeff(x)
        grid = self.embed_pod(coefficients.permute(0, 2, 3, 1))
        for layer in self.afno_layers:
            grid = layer(grid)
        predicted_coefficients = self.head_pod(grid).permute(0, 3, 1, 2)
        pod_flat = self.mean[None, None] + self._field(predicted_coefficients)
        return pod_flat.reshape(batch, self.output_steps, 2, height, width)

    def forward(self, x, return_aux=None):
        if x.ndim != 5 or x.shape[1] != self.input_steps or x.shape[2] != 2:
            raise ValueError(
                f"expected input [B, {self.input_steps}, 2, H, W], got {tuple(x.shape)}"
            )
        u_pod = self.forward_pod(x)

        if self.dual_stream_mode == "temporal_concat":
            x_hwc = x.permute(0, 1, 3, 4, 2)
            u_pod_hwc = u_pod.permute(0, 1, 3, 4, 2)
            dual_in = torch.cat([x_hwc, u_pod_hwc], dim=1)  # [B, T_in + T_out, H, W, 2]
            delta_all = self.residual_unet(dual_in)          # [B, T_in + T_out, H, W, 2]
            delta_hwc = self.time_project(delta_all.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        elif self.dual_stream_mode == "future_refine":
            u_pod_hwc = u_pod.permute(0, 1, 3, 4, 2)
            delta_hwc = self.residual_unet(u_pod_hwc)
        else:  # channel_concat
            dual_in = torch.cat(
                (x.permute(0, 1, 3, 4, 2), u_pod.permute(0, 1, 3, 4, 2)), dim=-1
            )
            delta_hwc = self.residual_unet(dual_in)

        delta_u = delta_hwc.permute(0, 1, 4, 2, 3)
        u_final = u_pod + delta_u
        ret_aux = self.return_aux if return_aux is None else return_aux
        return (u_final, u_pod) if ret_aux else u_final
