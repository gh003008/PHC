"""Pure-torch helpers for PHC-Pain-v0. No Isaac Gym, no numpy.

Shape legend: B = num_envs, D = num_dof (SMPL=69), Nb = num_bodies (SMPL=24).

Saturation notes (Spec #2 design §5.2):
- update_pain_state unclamped steady state for constant `inst` is
  12.5 * relu(inst - threshold) under rise=0.25, decay=0.02 defaults;
  sustained inst >= 0.34 drives pain_state to cap=3.0.
- apply_pain_action_guard saturates at pain_state >= 1.25 under
  guard_gain=0.6, max_guard=0.75 defaults; pain range [1.25, 3.0]
  collapses under the guard. Retune in Spec #3 if needed.
"""

import torch


def compute_joint_limit_pain(q, q_lo, q_hi, margin_ratio=0.15, eps=1e-6):
    """Pain rises when |q - q_mid| exceeds a safe band inside the joint ROM.

    Args:
        q:        (B, D) current DOF positions.
        q_lo:     (D,) lower limits.
        q_hi:     (D,) upper limits.
    Returns:
        (B, D) in [0, ~1/margin_ratio]. 0 inside the safe zone.
        Fixed joints (q_half <= eps) are masked to zero.
    """
    q_mid = 0.5 * (q_lo + q_hi)
    q_half = 0.5 * (q_hi - q_lo)
    q_safe = q_half * (1.0 - margin_ratio)
    excess = torch.relu(torch.abs(q - q_mid) - q_safe)
    pain = excess / (q_half - q_safe + eps)
    fixed_mask = (q_half > eps).to(pain.dtype)
    return pain * fixed_mask


def compute_torque_pain(tau, tau_limits, ref_scale=0.5, eps=1e-6):
    """|tau| normalized by a safe fraction of tau_limits, with a 1 Nm floor."""
    tau_ref = torch.clamp(ref_scale * tau_limits, min=1.0)
    return torch.abs(tau) / (tau_ref + eps)


def compute_power_pain(tau, dq, power_ref=5.0, eps=1e-6):
    """|tau * dq| normalized by a scalar power reference."""
    return torch.abs(tau * dq) / max(power_ref, eps)


def compute_contact_pain(contact_forces, ref_force=150.0, eps=1e-6):
    """Body-level contact force magnitude normalized by a scalar reference.

    Args:
        contact_forces: (B, Nb, 3).
    Returns:
        (B, Nb).
    """
    return torch.norm(contact_forces, dim=-1) / max(ref_force, eps)


def broadcast_body_pain_to_dof(body_pain, num_dof):
    """Map per-body scalars to per-DOF values assuming SMPL layout.

    SMPL: _dof_names == _body_names[1:], 3 consecutive DOFs per non-root body.
    Root body (index 0) has no DOFs and is dropped.

    Args:
        body_pain: (B, Nb).
        num_dof:   expected to equal 3 * (Nb - 1).
    Returns:
        (B, num_dof).
    """
    B, Nb = body_pain.shape
    assert num_dof == 3 * (Nb - 1), (
        f"broadcast_body_pain_to_dof expects SMPL layout num_dof==3*(Nb-1); "
        f"got num_dof={num_dof}, Nb={Nb}"
    )
    non_root = body_pain[:, 1:]
    return non_root.unsqueeze(-1).expand(B, Nb - 1, 3).reshape(B, -1)


def combine_internal_pain(
    limit_pain, torque_pain, power_pain,
    w_limit=1.0, w_torque=0.35, w_power=0.15,
):
    """Weighted sum of the three internal pain channels."""
    return w_limit * limit_pain + w_torque * torque_pain + w_power * power_pain


def update_pain_state(
    prev, inst,
    rise_alpha=0.25, decay_alpha=0.02, threshold=0.1, cap=3.0,
):
    """Leaky accumulator with threshold-gated drive and hard cap."""
    drive = torch.relu(inst - threshold)
    nxt = prev * (1.0 - decay_alpha) + rise_alpha * drive
    return torch.clamp(nxt, 0.0, cap)


def apply_pain_action_guard(
    pd_tar, dof_pos, pain_state,
    guard_gain=0.6, max_guard=0.75,
):
    """Contract PD target toward current dof_pos in proportion to pain.

    pd_tar and dof_pos must be in the same (radian DOF) space.
    """
    guard = torch.clamp(guard_gain * pain_state, 0.0, max_guard)
    return dof_pos + (1.0 - guard) * (pd_tar - dof_pos)
