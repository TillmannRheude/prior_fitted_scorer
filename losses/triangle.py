import torch
import torch.nn.functional as F

from losses.utils import apply_logit_scale


def compute_triangle_area_matrix(anchor, mod_b, mod_c, squared=True, eps=1e-8):
    """Return all areas between anchors and paired candidates from B and C."""
    u = anchor[:, None, :] - mod_b[None, :, :]
    v = anchor[:, None, :] - mod_c[None, :, :]
    determinant = (u.square().sum(-1) * v.square().sum(-1)) - (
        (u * v).sum(-1).square()
    )
    determinant = determinant.clamp_min(0.0)
    if squared:
        return 0.5 * determinant
    return 0.5 * torch.sqrt(determinant + eps)


def compute_triangle_area_matrix_efficient(
    anchor, mod_b, mod_c, squared=True, eps=1e-8
):
    """Compute the same area matrix without materializing a B x B x D tensor."""
    anchor_norm = anchor.square().sum(1, keepdim=True)
    b_norm = mod_b.square().sum(1).unsqueeze(0)
    c_norm = mod_c.square().sum(1).unsqueeze(0)
    anchor_b = anchor @ mod_b.T
    anchor_c = anchor @ mod_c.T
    paired_bc = (mod_b * mod_c).sum(1).unsqueeze(0)

    u_norm = anchor_norm + b_norm - 2.0 * anchor_b
    v_norm = anchor_norm + c_norm - 2.0 * anchor_c
    uv_dot = anchor_norm - anchor_b - anchor_c + paired_bc
    determinant = (u_norm * v_norm - uv_dot.square()).clamp_min(0.0)
    if squared:
        return 0.5 * determinant
    return 0.5 * torch.sqrt(determinant + eps)


def compute_triangle_infonce(
    anchor, mod_b, mod_c, logit_scale, squared=True, eps=1e-8
):
    area = compute_triangle_area_matrix_efficient(
        anchor, mod_b, mod_c, squared=squared, eps=eps
    )
    logits = apply_logit_scale(-area, logit_scale)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )


def triangle(r_a, r_b, r_c, logit_scale, **kwargs):
    squared = kwargs.get("squared", True)
    eps = kwargs.get("eps", 1e-8)
    losses = (
        compute_triangle_infonce(
            r_a, r_b, r_c, logit_scale, squared=squared, eps=eps
        ),
        compute_triangle_infonce(
            r_b, r_a, r_c, logit_scale, squared=squared, eps=eps
        ),
        compute_triangle_infonce(
            r_c, r_a, r_b, logit_scale, squared=squared, eps=eps
        ),
    )
    return sum(losses) / len(losses)
