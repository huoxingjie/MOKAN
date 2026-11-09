"""
HiPPO-LegS: High-order Polynomial Projection Operators (Legendre Scaled measure).

Implements orthonormal Legendre polynomial basis projection for time series.
- encode:  [B, T, F] -> [B, N, F]  (compress time dimension to N Legendre coefficients)
- decode:  [B, N, F] -> [B, T, F]  (reconstruct time dimension from coefficients)

Reference: Gu et al., "HiPPO: Recurrent Memory with Optimal Polynomial Projections", NeurIPS 2020.
"""

import torch
import torch.nn as nn
import math


def _compute_legendre_basis(T: int, N: int, device: torch.device, dtype: torch.dtype = torch.float32):
    """
    Compute orthonormal Legendre basis functions for T discrete time steps.

    Basis functions are evaluated on the interval [-1, 1] with T uniformly-spaced points.
    Uses the three-term recurrence relation for Legendre polynomials.

    Args:
        T: number of time steps
        N: number of basis functions (polynomial degree 0 through N-1)
        device: torch device
        dtype: torch dtype

    Returns:
        basis: [N, T] tensor of orthonormal Legendre basis values
    """
    if T <= 1:
        return torch.ones(1, T, device=device, dtype=dtype)

    t = torch.linspace(-1.0, 1.0, T, device=device, dtype=dtype)
    basis_list = []

    # P_0(x) = 1,  normalized: sqrt(1/2)
    p_prev = torch.ones_like(t)
    basis_list.append(p_prev * math.sqrt(0.5))

    if N == 1:
        return torch.stack(basis_list, dim=0)

    # P_1(x) = x,  normalized: sqrt(3/2)
    p_curr = t.clone()
    basis_list.append(p_curr * math.sqrt(1.5))

    for n in range(2, N):
        p_next = ((2.0 * n - 1.0) * t * p_curr - (n - 1.0) * p_prev) / n
        norm = math.sqrt(n + 0.5)
        basis_list.append(p_next * norm)
        p_prev, p_curr = p_curr, p_next

    return torch.stack(basis_list, dim=0)


class HiPPOLegS(nn.Module):
    """
    HiPPO-LegS (Scaled Legendre) projection layer.
    Encodes a time series into Legendre polynomial coefficients and decodes back.

    Args:
        N: number of Legendre coefficients (polynomial degree)
    """

    def __init__(self, N: int = 32):
        super().__init__()
        self.N = N

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, T, F = x.shape
            basis = _compute_legendre_basis(T, self.N, x.device, x.dtype)
            dx = 2.0 / (T - 1) if T > 1 else 1.0
            c = torch.einsum('nt,btf->bnf', basis, x) * dx
            return c
        elif x.dim() == 2:
            B, T = x.shape
            basis = _compute_legendre_basis(T, self.N, x.device, x.dtype)
            dx = 2.0 / (T - 1) if T > 1 else 1.0
            c = torch.einsum('nt,bt->bn', basis, x) * dx
            return c
        else:
            raise ValueError(f"Expected 2 or 3 dims, got {x.dim()}")

    def decode(self, c: torch.Tensor, T_out: int) -> torch.Tensor:
        basis = _compute_legendre_basis(T_out, self.N, c.device, c.dtype)
        if c.dim() == 3:
            x = torch.einsum('nt,bnf->btf', basis, c)
        elif c.dim() == 2:
            x = torch.einsum('nt,bn->bt', basis, c)
        else:
            raise ValueError(f"Expected 2 or 3 dims, got {c.dim()}")
        return x

    def forward(self, x: torch.Tensor, T_out: int = None):
        c = self.encode(x)
        if T_out is None:
            T_out = x.shape[-2] if x.dim() == 3 else x.shape[-1]
        return self.decode(c, T_out)


class HiPPOFeatureTransform(nn.Module):
    """
    HiPPO-based feature dimension transformation.

    Flow:
    1. HiPPO encode: [B, T, F_in] -> [B, N, F_in]
    2. Feature projection: [B, N, F_in] -> [B, N, F_out]
    3. HiPPO decode: [B, N, F_out] -> [B, T_out, F_out]

    Args:
        in_features: input feature dimension
        out_features: target feature dimension
        N: number of Legendre coefficients
    """

    def __init__(self, in_features: int, out_features: int, N: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.N = N
        self.hippo = HiPPOLegS(N=N)
        self.feature_proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, T_out: int = None) -> torch.Tensor:
        B, T, F_in = x.shape
        if T_out is None:
            T_out = T

        # Step 1: encode
        c = self.hippo.encode(x)  # [B, N, F_in]

        # Step 2: feature projection
        c = self.feature_proj(c.reshape(B * self.N, F_in))  # [B*N, F_out]
        c = c.view(B, self.N, self.out_features)  # [B, N, F_out]

        # Step 3: decode
        y = self.hippo.decode(c, T_out)  # [B, T_out, F_out]
        return y


class HiPPOTimeBranch(nn.Module):
    """
    HiPPO-based temporal branch for EnsembleKAN.
    Replaces nn.Linear(in_length, out_length) with HiPPO encode-decode.

    Args:
        in_length: input sequence length
        out_length: output sequence length
        N: number of Legendre coefficients
    """

    def __init__(self, in_length: int, out_length: int, N: int = 32):
        super().__init__()
        self.in_length = in_length
        self.out_length = out_length
        self.hippo = HiPPOLegS(N=N)
        self.proj = nn.Linear(N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        c = self.hippo.encode(x)  # [B, N, F]
        c = c.permute(0, 2, 1).contiguous()  # [B, F, N]
        c = self.proj(c)  # [B, F, N]
        c = c.permute(0, 2, 1).contiguous()  # [B, N, F]
        y = self.hippo.decode(c, self.out_length)  # [B, T_out, F]
        return y
