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


class ResAMPVCmdNetwork(nn.Module):
    """Wraps frozen phc_3 PNN + ResMLPHead. Output = a_base + clip(Δa, ±0.2).

    The frozen base (phc_3) is loaded externally from a state_dict. This wrapper
    just provides the forward path expected by rl_games' AmpAgent.
    """

    def __init__(self, frozen_base: nn.Module, residual_head: ResMLPHead,
                 delta_clip: float = 0.2):
        super().__init__()
        self.frozen_base = frozen_base
        for p in self.frozen_base.parameters():
            p.requires_grad = False
        self.frozen_base.eval()
        self.residual_head = residual_head
        self.delta_clip = float(delta_clip)

    def forward(self, obs: torch.Tensor, ref: torch.Tensor, v_cmd: torch.Tensor,
                proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, sigma) for action distribution."""
        with torch.no_grad():
            a_base, _ = self.frozen_base(obs, ref)   # (N, 69)
        # Residual: input is (proprio, v_cmd_norm)
        x = torch.cat([proprio, v_cmd.unsqueeze(-1) if v_cmd.dim() == 1 else v_cmd], dim=-1)
        delta_mu, delta_sigma = self.residual_head(x)
        # Clip Δa to keep residual bounded
        delta_mu = torch.clamp(delta_mu, -self.delta_clip, self.delta_clip)
        mu = a_base + delta_mu
        # Sigma comes from residual only (base is deterministic given ref).
        return mu, delta_sigma


if __name__ == "__main__":
    head = ResMLPHead(in_dim=246, action_dim=69)
    x = torch.randn(2, 246)
    mu1, sigma1 = head(x)
    assert mu1.shape == (2, 69), mu1.shape
    assert sigma1.shape == (2, 69), sigma1.shape
    assert sigma1.abs().min() > 0
    n_params = sum(p.numel() for p in head.parameters())
    print(f"OK ResMLPHead, params={n_params}")

    # Mock frozen base: a deterministic identity-ish stub
    class MockBase(nn.Module):
        def forward(self, obs, ref):
            return torch.zeros(obs.shape[0], 69), None
    base = MockBase()
    net = ResAMPVCmdNetwork(frozen_base=base, residual_head=head, delta_clip=0.2)
    obs2 = torch.randn(4, 1000)
    ref2 = torch.randn(4, 800)
    proprio = torch.randn(4, 245)
    v_cmd = torch.randn(4)
    mu2, sigma2 = net(obs2, ref2, v_cmd, proprio)
    assert mu2.shape == (4, 69)
    assert sigma2.shape == (4, 69)
    assert mu2.abs().max() <= 0.2 + 1e-6, "Δa should be clipped"
    for p in net.frozen_base.parameters():
        assert not p.requires_grad
    print("OK ResAMPVCmdNetwork")
