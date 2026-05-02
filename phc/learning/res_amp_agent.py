"""Residual + AMP training agent — inherits PHC's IMAmpAgent, freezes phc_3 base.

Spec: docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md
"""
from __future__ import annotations

# Match the dual-path import style of run.py / run_hydra.py: those scripts add
# `phc/` to sys.path, so `from learning.im_amp import ...` works at runtime,
# but it also works as `phc.learning.im_amp` for standalone imports.
try:
    from learning.im_amp import IMAmpAgent
except ImportError:  # pragma: no cover - fallback for non-runner contexts
    from phc.learning.im_amp import IMAmpAgent


class ResAMPVCmdAgent(IMAmpAgent):
    """AMP agent where only the residual head + critic + AMP discriminator
    are trainable. The phc_3 PNN actor parameters are frozen at load time."""

    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        # Freeze the frozen_base submodule of the network
        net = self.model.a2c_network
        if hasattr(net, "frozen_base") or hasattr(net, "pnn"):
            # Task 13's network swaps `self.pnn` for the loaded frozen base
            base = getattr(net, "frozen_base", None) or getattr(net, "pnn", None)
            if base is not None:
                for p in base.parameters():
                    p.requires_grad = False
                base.eval()
                n_frozen = sum(p.numel() for p in base.parameters())
                n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print(f"[ResAMPVCmdAgent] frozen={n_frozen}  trainable={n_train}")
            else:
                print("[ResAMPVCmdAgent] WARN: net has neither .frozen_base nor .pnn — base not frozen")
        else:
            print("[ResAMPVCmdAgent] WARN: net has no .frozen_base or .pnn attribute")

        # Apply phc_3 RunningMeanStd to agent's running_mean_std (mirrors the
        # IMAMPPlayer hook, but for training mode). Without this, training
        # starts with default-init RMS (zeros mean, unit var) → phc_3 receives
        # garbage-normalized obs → falls in 5 step within first epoch (verified).
        # Stats will continue updating during rollout, but the warm-start from
        # phc_3's stats keeps phc_3 walking from epoch 1.
        import torch
        frozen_rms = getattr(net, "_frozen_rms_state", None)
        if (frozen_rms is not None and getattr(self, "normalize_input", False)
                and getattr(self, "running_mean_std", None) is not None):
            try:
                phc3_mean = frozen_rms["running_mean"]
                phc3_var = frozen_rms["running_var"]
                cur_dim = self.running_mean_std.running_mean.shape[0]
                if phc3_mean.shape[0] == cur_dim:
                    target_mean, target_var = phc3_mean, phc3_var
                elif phc3_mean.shape[0] == cur_dim - 1:
                    pad_mean = torch.zeros(1, dtype=phc3_mean.dtype, device=phc3_mean.device)
                    pad_var = torch.ones(1, dtype=phc3_var.dtype, device=phc3_var.device)
                    target_mean = torch.cat([phc3_mean, pad_mean])
                    target_var = torch.cat([phc3_var, pad_var])
                else:
                    target_mean = None
                if target_mean is not None:
                    self.running_mean_std.running_mean.data.copy_(target_mean.to(self.running_mean_std.running_mean.device))
                    self.running_mean_std.running_var.data.copy_(target_var.to(self.running_mean_std.running_var.device))
                    self.running_mean_std.count.data.copy_(frozen_rms["count"].to(self.running_mean_std.count.device))
                    print(f"[ResAMPVCmdAgent] applied phc_3 RunningMeanStd ({target_mean.shape[0]}-D) — training warm-start")
            except Exception as _e:
                print(f"[ResAMPVCmdAgent] failed to apply phc_3 RMS: {_e}")

    def calc_gradients(self, input_dict):
        """Override to assert phc_3 base never gets gradients."""
        super().calc_gradients(input_dict)
        # Sanity: frozen base must not have grads
        net = self.model.a2c_network
        base = getattr(net, "frozen_base", None) or getattr(net, "pnn", None)
        if base is not None:
            for n, p in base.named_parameters():
                if p.grad is not None and p.grad.abs().sum().item() > 0:
                    raise RuntimeError(
                        f"[ResAMPVCmdAgent] frozen_base param {n} got non-zero grad!")
