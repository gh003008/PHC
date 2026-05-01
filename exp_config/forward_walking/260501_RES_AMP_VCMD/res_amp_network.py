"""Residual + AMP network for v_cmd-conditioned policy on top of frozen phc_3.

Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md

Three layers:
  * ``ResMLPHead`` — small (proprio + v_cmd) → Δa MLP, fixed sigma.
  * ``ResAMPVCmdNetwork`` — wraps a frozen phc_3 PNN + ``ResMLPHead`` and
    speaks rl_games' continuous-AMP network protocol (forward returns
    ``(mu, logstd, value, states)`` and exposes ``eval_disc``).
  * ``ResAMPVCmdBuilder`` — rl_games-style network builder that the
    learning yaml's ``params.network.name = amp_pnn_residual`` resolves to.

The forward path takes the standard rl_games ``input_dict``:

  obs_dict['obs']  -> [phc_3 obs (934-D) | v_cmd (1-D)]   (full obs vector)

The frozen base reads obs[:, :-v_cmd_dim] (i.e. drop v_cmd → original 934-D);
the residual head reads the full obs (proprio is just everything before v_cmd
in PHC's flat obs, and we additionally append v_cmd so the head sees velocity
context). This keeps the seam between Tasks 11 and 13 minimal.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from phc.learning.amp_network_pnn_builder import AMPPNNBuilder


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
        # log_sigma is a (non-learnable by default) parameter so it survives
        # state_dict round-trips and lives on the right device automatically.
        self.log_sigma = nn.Parameter(torch.full((action_dim,), float(sigma_init)),
                                       requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        logstd = self.log_sigma.expand_as(mu)
        return mu, logstd


class ResAMPVCmdNetwork(AMPPNNBuilder.Network):
    """Frozen phc_3 PNN + residual head, with full AMP critic/disc machinery.

    We subclass ``AMPPNNBuilder.Network`` to inherit:
      - ``forward(obs_dict)`` (from AMPBuilder) glue between actor/critic
      - ``eval_critic`` (separate MLP, built from yaml's mlp.units)
      - ``_build_disc`` / ``eval_disc``
      - sigma parameter setup

    We override ``__init__`` to swap the actor path (replacing the trainable
    PNN with a *frozen* phc_3 PNN) and add the residual head. We override
    ``eval_actor`` to compute ``a_base + clip(Δa, ±0.2)``.

    Note: the wrapping ``a2c_network.mu`` head from rl_games is unused
    (phc_3 PNN columns already terminate in a 69-D linear) — we leave it
    constructed by the parent for state_dict compatibility but ignore it.
    """

    def __init__(self, params, **kwargs):
        # AMPPNNBuilder.Network.__init__ will:
        #   1. read self_obs_size / task_obs_size / task_obs_size_detail kwargs
        #   2. overwrite kwargs['input_shape'] (in ITS local kwargs only) to
        #      (self_obs_size + task_obs_size,) — does NOT mutate our kwargs
        #      because **kwargs spreads into a fresh dict per call
        #   3. construct a fresh trainable PNN (we'll discard it below)
        #   4. build critic_mlp with the (mutated, smaller) input dim → this
        #      misses the v_cmd dim our env appends to obs. We rebuild below.
        super().__init__(params, **kwargs)

        # ---- Rebuild critic_mlp to accept full obs (incl. v_cmd) ----
        # Parent built critic with input_size = self_obs + task_obs (934 for
        # phc_3 layout). Our env's get_obs_size() = self_obs + task_obs + 1
        # (v_cmd appended), so the env feeds the network 935-D obs at runtime,
        # and parent's critic_mlp.0 expects 934 → matmul mismatch.
        # Rebuild with full obs dim so V(s) is v_cmd-aware and the matmul
        # matches at the env→network boundary.
        residual_cfg_for_vcmd = params.get("residual", {}) if isinstance(params, dict) else {}
        v_cmd_dim_for_rebuild = int(residual_cfg_for_vcmd.get("v_cmd_dim", 1))
        critic_in = self.self_obs_size + self.task_obs_size + v_cmd_dim_for_rebuild
        critic_mlp_args = {
            'input_size': critic_in,
            'units': self.units,
            'activation': self.activation,
            'norm_func_name': self.normalization,
            'dense_func': torch.nn.Linear,
            'd2rl': self.is_d2rl,
            'norm_only_first_layer': self.norm_only_first_layer,
        }
        self.critic_mlp = self._build_mlp(**critic_mlp_args)
        mlp_init = self.init_factory.create(**self.initializer)
        for m in self.critic_mlp.modules():
            if isinstance(m, torch.nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        # ---- Replace trainable PNN with frozen phc_3 ----
        from phc.learning.res_amp_network import load_frozen_phc_3  # self-import for clarity
        frozen_cfg = params.get("frozen_base", {}) if isinstance(params, dict) else {}
        ckpt_path = frozen_cfg.get("checkpoint", "output/HumanoidIm/phc_3/Humanoid.pth")
        # Load to CPU here; the rl_games framework will .to(device) the whole net later.
        frozen = load_frozen_phc_3(ckpt_path, device="cpu")
        for p in frozen.parameters():
            p.requires_grad = False
        # Discard the freshly-initialized self.pnn (built by AMPPNNBuilder.Network)
        # and substitute the frozen one. We keep the attribute name `pnn`
        # so any downstream PHC code that references `network.pnn` still works.
        del self.pnn
        self.pnn = frozen
        # phc_3 was trained with `training_prim` final column index — the
        # checkpoint's last actor (index num_prim-1) is the highest-quality one.
        # We use the top column for inference; if yaml overrides, honor it.
        self.training_prim = int(frozen_cfg.get("num_actors", getattr(self, "num_prim", 3))) - 1

        # ---- Build residual head ----
        residual_cfg = params.get("residual", {}) if isinstance(params, dict) else {}
        # Residual input = full obs vector (proprio + v_cmd, in PHC flat layout)
        action_dim = kwargs["actions_num"]
        in_dim = kwargs["input_shape"][0]
        hidden = tuple(residual_cfg.get("mlp_units", [256, 256]))
        sigma_init = float(residual_cfg.get("sigma_init", -1.0))
        self.residual_head = ResMLPHead(
            in_dim=in_dim,
            action_dim=action_dim,
            hidden=hidden,
            sigma_init=sigma_init,
        )

        self.delta_clip = float(params.get("delta_clip", 0.2)) if isinstance(params, dict) else 0.2

        # v_cmd dim — assumed last entry of the obs vector. The env
        # extends obs by 1 (Task 6: v_cmd observation extension).
        self._v_cmd_dim = int(residual_cfg.get("v_cmd_dim", 1))

    # ---- Actor path override ----
    def eval_actor(self, obs_dict):
        obs = obs_dict['obs']
        # Frozen base sees the original phc_3 obs (drop v_cmd tail)
        base_obs = obs[:, : obs.shape[-1] - self._v_cmd_dim] if self._v_cmd_dim > 0 else obs
        # PNN inference: returns (final_col_out, [all_cols_out])
        with torch.no_grad():
            base_in = self.actor_cnn(base_obs)
            base_in = base_in.contiguous().view(base_in.size(0), -1)
            a_base, _ = self.pnn(base_in, idx=self.training_prim)

        # Residual head sees the full obs (proprio + v_cmd)
        delta_mu, delta_logstd = self.residual_head(obs)
        delta_mu = torch.clamp(delta_mu, -self.delta_clip, self.delta_clip)

        mu = a_base + delta_mu

        if self.is_continuous:
            # Use residual's own logstd (base is deterministic given obs).
            return mu, delta_logstd

        # Non-continuous paths: defer to parent (won't actually be hit in PHC).
        return super().eval_actor(obs_dict)


class ResAMPVCmdBuilder(AMPPNNBuilder):
    """rl_games-style network builder.

    Reads cfg from the learning yaml's ``params.network`` block (passed in
    via ``load(params)``) and returns a ``ResAMPVCmdNetwork`` instance.
    """

    def build(self, name, **kwargs):
        return ResAMPVCmdNetwork(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)


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
    n_params = sum(p.numel() for p in head.parameters())
    print(f"OK ResMLPHead, params={n_params}")

    # Builder smoke (no actual ckpt → just import-ability)
    b = ResAMPVCmdBuilder()
    print(f"OK ResAMPVCmdBuilder instance: {b}")

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
