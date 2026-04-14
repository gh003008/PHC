"""Stability margin (alpha) utilities for the Motion Plan Layer.

These functions operate on plain PyTorch tensors and have no IsaacGym
dependency, so they are trivially unit-testable and reusable across tasks.

Alpha is a continuous stability margin in [0, 1]:
    0 -> very stable (XCoM well inside support polygon)
    1 -> critically unstable (XCoM at or outside support polygon boundary)

The concept document (260410_motion_plan_layer_concept_v09.docx §3.1, and
docs/260410_motion_plan_layer_issues_and_extensions.md) specifies alpha as
the single scalar that drives continuous task-following ↔ recovery blending,
avoiding binary mode switching and its discontinuities.

Public API:
    compute_cop(contact_forces, contact_positions)
    compute_support_polygon_rect(contact_positions, contact_flags)
    compute_xcom(com_pos, com_vel, com_height, g=9.81)
    compute_alpha(xcom_xy, polygon_rect)
"""

from typing import Tuple

import torch


# --------------------------------------------------------------- CoP

def compute_cop(
    contact_forces: torch.Tensor,
    contact_positions: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Batched Center of Pressure from per-foot contact forces.

    Args:
        contact_forces: [N, C, 3] vertical component (z) carries the load.
        contact_positions: [N, C, 3] world-frame XY positions of each contact body.
        eps: numerical guard on zero total vertical force.

    Returns:
        cop_xy: [N, 2] center-of-pressure in world XY. When no foot is loaded,
            this defaults to the mean contact position so that downstream alpha
            values remain well-defined (alpha becomes irrelevant in flight anyway).
    """
    fz = contact_forces[..., 2].clamp(min=0.0)                       # [N, C]
    xy = contact_positions[..., :2]                                  # [N, C, 2]
    total_fz = fz.sum(dim=-1, keepdim=True).clamp(min=eps)           # [N, 1]
    cop_xy = (xy * fz.unsqueeze(-1)).sum(dim=1) / total_fz           # [N, 2]

    # Fallback for flight phases: use mean contact position.
    in_flight = (fz.sum(dim=-1) < eps)                               # [N]
    if in_flight.any():
        mean_xy = xy.mean(dim=1)                                     # [N, 2]
        cop_xy = torch.where(in_flight.unsqueeze(-1), mean_xy, cop_xy)
    return cop_xy


# --------------------------------------------------------- Support polygon

def compute_support_polygon_rect(
    contact_positions: torch.Tensor,
    contact_flags: torch.Tensor,
    margin: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Axis-aligned bounding rectangle of the grounded feet.

    A proper convex hull is overkill for 2-4 feet; the AABB is a fast and
    monotonic approximation that works well for forward walking.

    Args:
        contact_positions: [N, C, 3]
        contact_flags: [N, C] bool/float, 1 where the foot is grounded.
        margin: meters of inward shrink/outward expansion of the rectangle
            to account for finite foot size (+) or safety gap (-).

    Returns:
        (polygon_min_xy, polygon_max_xy): each [N, 2].
        For envs in flight (no grounded foot) the rectangle collapses to
        a zero-area box at the centroid — alpha = 1 in that case, by design.
    """
    xy = contact_positions[..., :2]                                  # [N, C, 2]
    flag = contact_flags.unsqueeze(-1).float()                       # [N, C, 1]

    # Mask: grounded rows get their XY, ungrounded get +inf / -inf so they
    # don't affect min/max. Then fallback to centroid for all-flight envs.
    INF = 1e6
    masked_min = xy * flag + (1.0 - flag) * INF
    masked_max = xy * flag + (1.0 - flag) * (-INF)

    poly_min = masked_min.min(dim=1).values                          # [N, 2]
    poly_max = masked_max.max(dim=1).values                          # [N, 2]

    in_flight = (contact_flags.sum(dim=-1) < 0.5)                    # [N] bool
    if in_flight.any():
        centroid = xy.mean(dim=1)                                    # [N, 2]
        poly_min = torch.where(in_flight.unsqueeze(-1), centroid, poly_min)
        poly_max = torch.where(in_flight.unsqueeze(-1), centroid, poly_max)

    # Expand by margin (positive = outward, negative = inward).
    poly_min = poly_min - margin
    poly_max = poly_max + margin
    return poly_min, poly_max


# ---------------------------------------------------------------- XCoM

def compute_xcom(
    com_pos: torch.Tensor,
    com_vel: torch.Tensor,
    com_height: torch.Tensor,
    g: float = 9.81,
    h_min: float = 0.3,
) -> torch.Tensor:
    """Extrapolated Center of Mass (Hof, 2008).

        xcom_xy = com_xy + com_vel_xy / sqrt(g / h)

    Args:
        com_pos: [N, 3] world-frame CoM position.
        com_vel: [N, 3] world-frame CoM velocity.
        com_height: [N] effective pendulum height (CoM z minus ground).
            Clamped below by `h_min` to avoid division blow-ups in falls.

    Returns:
        xcom_xy: [N, 2]
    """
    h = com_height.clamp(min=h_min)                                  # [N]
    omega = torch.sqrt(torch.tensor(g, device=com_pos.device) / h)    # [N]
    xcom = com_pos[..., :2] + com_vel[..., :2] / omega.unsqueeze(-1)
    return xcom


# --------------------------------------------------------------- Alpha

def compute_alpha(
    xcom_xy: torch.Tensor,
    polygon_min_xy: torch.Tensor,
    polygon_max_xy: torch.Tensor,
    scale: float = 0.2,
) -> torch.Tensor:
    """Stability margin in [0, 1] from XCoM position relative to the AABB polygon.

    Signed distance: positive when XCoM is inside, negative when outside.
    `alpha` maps this signed distance through a smooth sigmoid-like function:

        d_inside = min(distance to each polygon edge, clamped >= 0)   # inside
        d_outside = distance_to_polygon                               # outside (>=0)
        signed = d_inside if inside else -d_outside
        alpha = clamp(1 - signed / scale, 0, 1)

    So XCoM at least `scale` meters inside -> alpha = 0 (fully stable).
    XCoM on the boundary -> alpha = 1 (critical).
    XCoM outside -> alpha saturates at 1.

    Args:
        xcom_xy: [N, 2]
        polygon_min_xy: [N, 2]
        polygon_max_xy: [N, 2]
        scale: meters — the "safety buffer" distance that maps to alpha 0→1.

    Returns:
        alpha: [N]
    """
    # Distance from xcom to each polygon edge (positive = inside that edge).
    to_left = xcom_xy[..., 0] - polygon_min_xy[..., 0]
    to_right = polygon_max_xy[..., 0] - xcom_xy[..., 0]
    to_back = xcom_xy[..., 1] - polygon_min_xy[..., 1]
    to_front = polygon_max_xy[..., 1] - xcom_xy[..., 1]

    edges = torch.stack([to_left, to_right, to_back, to_front], dim=-1)  # [N, 4]

    # Inside if all edges non-negative.
    inside = (edges.min(dim=-1).values >= 0)                             # [N]

    # Signed "radius" inside polygon (smallest distance to any edge).
    inside_margin = edges.min(dim=-1).values.clamp(min=0.0)              # [N]

    # Outside distance: Euclidean overshoot to the nearest edge.
    over_x = torch.clamp(-to_left, min=0.0) + torch.clamp(-to_right, min=0.0)
    over_y = torch.clamp(-to_back, min=0.0) + torch.clamp(-to_front, min=0.0)
    outside_dist = torch.sqrt(over_x * over_x + over_y * over_y + 1e-12)

    signed = torch.where(inside, inside_margin, -outside_dist)
    alpha = torch.clamp(1.0 - signed / scale, 0.0, 1.0)
    return alpha


# --------------------------------------------------------------- combined

def compute_stability(
    contact_forces: torch.Tensor,
    contact_positions: torch.Tensor,
    com_pos: torch.Tensor,
    com_vel: torch.Tensor,
    contact_threshold: float = 10.0,
    polygon_margin: float = 0.05,
    alpha_scale: float = 0.2,
    ground_height: float = 0.0,
) -> dict:
    """One-shot convenience wrapper: all the pieces in a single call.

    Returns:
        dict with keys: 'cop', 'xcom', 'polygon_min', 'polygon_max', 'alpha'
    """
    fz = contact_forces[..., 2].clamp(min=0.0)
    contact_flags = (fz > contact_threshold).float()

    cop = compute_cop(contact_forces, contact_positions)
    poly_min, poly_max = compute_support_polygon_rect(
        contact_positions, contact_flags, margin=polygon_margin
    )
    h_com = (com_pos[..., 2] - ground_height).clamp(min=0.3)
    xcom = compute_xcom(com_pos, com_vel, h_com)
    alpha = compute_alpha(xcom, poly_min, poly_max, scale=alpha_scale)

    return {
        "cop": cop,
        "xcom": xcom,
        "polygon_min": poly_min,
        "polygon_max": poly_max,
        "alpha": alpha,
    }
