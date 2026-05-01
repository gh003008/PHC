"""Residual + AMP network for v_cmd-conditioned policy on top of frozen phc_3.

Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ResMLPHead(nn.Module):
    """Small residual MLP: (proprio + v_cmd_norm) -> Δa (69-D).

    Architecture per spec §3:
      Linear(in_dim, 256) → SiLU → Linear(256, 256) → SiLU → Linear(256, action_dim)
    Sigma is parametric (learn_sigma=False, fixed sigma_init=-1.0 → std=0.37).
    """

    def __init__(self, in_dim: int, action_dim: int = 69,
                 hidden: tuple[int, int] = (256, 256),
                 sigma_init: float = -1.0):
        super().__init__()
        self.action_dim = action_dim
        self.mu = nn.Sequential(
            nn.Linear(in_dim, hidden[0]),
            nn.SiLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.SiLU(),
            nn.Linear(hidden[1], action_dim),
        )
        self.log_sigma = nn.Parameter(torch.full((action_dim,), float(sigma_init)),
                                       requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        sigma = self.log_sigma.exp().expand_as(mu)
        return mu, sigma


if __name__ == "__main__":
    head = ResMLPHead(in_dim=246, action_dim=69)
    x = torch.randn(2, 246)
    mu, sigma = head(x)
    assert mu.shape == (2, 69), mu.shape
    assert sigma.shape == (2, 69), sigma.shape
    assert sigma.abs().min() > 0
    n_params = sum(p.numel() for p in head.parameters())
    print(f"OK ResMLPHead, params={n_params}")
