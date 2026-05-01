"""Residual + AMP network for v_cmd-conditioned policy on top of frozen phc_3.

Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def load_frozen_phc_3(checkpoint_path: str, device: str = "cuda:0") -> nn.Module:
    """Load phc_3 PNN from checkpoint and return a frozen nn.Module.

    The checkpoint is the standard rl_games / PHC im_pnn_big format:
      - state_dict under 'model' key
      - PNN keys at `a2c_network.pnn.actors.{0,1,2}.{0,2,4,...}.{weight,bias}`
      - phc_3 was trained with input_shape=(934,), action_dim=69, num_prim=3,
        has_lateral=False, units=[2048,1536,1024,1024,512,512], silu activation

    Notes on the wrapping `mu` head:
      Unlike the non-PNN PHC checkpoints, phc_3 stores `a2c_network.mu.{weight,bias}`
      separately from the PNN actors (each PNN column already terminates in a
      69-D linear layer). The PNN forward returns the column's final 69-D output
      directly, so we ignore the top-level `a2c_network.mu` here — that head is
      only used by the rl_games training-time wrapper, not at inference of the
      frozen base. Sigma is also unused (Δa adds its own).

    Returns: frozen `PNN` module on `device`, in eval mode, requires_grad=False.
    """
    try:
        # Prefer relative imports (PHC's runtime usually has phc/ on PYTHONPATH).
        from phc.learning.pnn import PNN  # type: ignore
    except ImportError:  # pragma: no cover - fallback for direct script run
        import os, sys
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.dirname(os.path.dirname(here)))
        from phc.learning.pnn import PNN  # type: ignore

    # torch.load: weights_only=False is needed because the checkpoint contains
    # rl_games' RunningMeanStd objects under top-level keys. Some older torch
    # versions don't accept that kwarg, so fall back gracefully.
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)

    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Discover PNN dims directly from the state_dict so we don't drift from
    # whatever phc_3 actually was. Use actor 0 as reference.
    actor0_keys = sorted(
        [k for k in sd.keys() if k.startswith("a2c_network.pnn.actors.0.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[-2]),
    )
    if not actor0_keys:
        raise RuntimeError(
            f"No PNN actor.0 keys found in {checkpoint_path} — is this a phc_3-style PNN ckpt?"
        )
    input_size = sd[actor0_keys[0]].shape[1]
    output_size = sd[actor0_keys[-1]].shape[0]
    units = [sd[k].shape[0] for k in actor0_keys[:-1]]  # all but the final 69-D linear

    num_prim = len({k.split(".")[3] for k in sd.keys() if k.startswith("a2c_network.pnn.actors.")})
    has_lateral = any(k.startswith("a2c_network.pnn.u") for k in sd.keys())

    mlp_args = {
        "input_size": input_size,
        "units": units,
        "activation": "silu",
        "dense_func": torch.nn.Linear,
    }
    pnn = PNN(mlp_args, output_size=output_size, numCols=num_prim, has_lateral=has_lateral)

    # Strip the `a2c_network.pnn.` prefix → matches PNN's own state_dict keys.
    pnn_sd_target = pnn.state_dict()
    loaded = 0
    for k, v in sd.items():
        if not k.startswith("a2c_network.pnn."):
            continue
        sub_key = k[len("a2c_network.pnn."):]
        if sub_key in pnn_sd_target:
            pnn_sd_target[sub_key].copy_(v)
            loaded += 1
    print(f"[load_frozen_phc_3] loaded {loaded}/{len(pnn_sd_target)} PNN tensors "
          f"(num_prim={num_prim}, has_lateral={has_lateral}, "
          f"input={input_size}, output={output_size}, units={units})")

    pnn.freeze_pnn(num_prim)
    for p in pnn.parameters():
        p.requires_grad = False
    pnn = pnn.to(device).eval()
    return pnn


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

    import os
    ckpt = "output/HumanoidIm/phc_3/Humanoid.pth"
    if os.path.exists(ckpt):
        print(f"[loader test] loading {ckpt} ...")
        try:
            base_pnn = load_frozen_phc_3(ckpt, device="cpu")
            n_p = sum(p.numel() for p in base_pnn.parameters())
            print(f"OK load_frozen_phc_3, params={n_p}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"WARN loader test failed: {e}")
    else:
        print(f"[loader test] skipped (no {ckpt})")
