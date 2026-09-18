import torch
import torch.nn.functional as F
from losses.utils import apply_logit_scale


def compute_triangle_area_matrix(anchor, mod_b, mod_c, squared=True, eps=1e-8):
    """
    Computes the pairwise TRIANGLE area matrix between:
      - anchor[i]
      - mod_b[j]
      - mod_c[j]

    for all i, j in the batch.

    Args:
        anchor, mod_b, mod_c (torch.Tensor): Shape (batch_sz, d).
            Assumed to already live in the shared embedding space.
            If your pipeline uses embedding_norm=True, these are typically L2-normalized.
        squared (bool): If True, returns the squared area surrogate used in the
            official repo/README example (no sqrt). If False, returns the actual area.
        eps (float): Numerical stability epsilon for the sqrt case.

    Returns:
        area (torch.Tensor): Shape (batch_sz, batch_sz), where smaller is better.
    """
    # anchor[i] paired against all mod_b[j], mod_c[j]
    anchor_exp = anchor.unsqueeze(1)      # (B, 1, D)
    mod_b_exp = mod_b.unsqueeze(0)        # (1, B, D)
    mod_c_exp = mod_c.unsqueeze(0)        # (1, B, D)

    u = anchor_exp - mod_b_exp            # (B, B, D)
    v = anchor_exp - mod_c_exp            # (B, B, D)

    u_norm = torch.sum(u * u, dim=-1)     # (B, B)
    v_norm = torch.sum(v * v, dim=-1)     # (B, B)
    uv_dot = torch.sum(u * v, dim=-1)     # (B, B)

    # For two sides u, v of the triangle:
    # area = 0.5 * sqrt(||u||^2 ||v||^2 - <u, v>^2)
    det_term = u_norm * v_norm - uv_dot * uv_dot
    det_term = torch.clamp(det_term, min=0.0)

    if squared:
        # Matches the official repo's practical implementation:
        # they drop the sqrt and keep the area surrogate.
        return 0.5 * det_term
    else:
        return 0.5 * torch.sqrt(det_term + eps)


def compute_triangle_area_matrix_efficient(anchor, mod_b, mod_c, squared=True, eps=1e-8):
    """
    Memory-efficient equivalent of `compute_triangle_area_matrix` that avoids
    materializing (B, B, D) tensors.

    It uses the identities:
      ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
      (a - b)^T(a - c) = ||a||^2 - <a, b> - <a, c> + <b, c>

    Args:
        anchor, mod_b, mod_c (torch.Tensor): Shape (batch_sz, d).
        squared (bool): If True, returns the squared area surrogate used in the
            official repo/README example (no sqrt). If False, returns the actual area.
        eps (float): Numerical stability epsilon for the sqrt case.

    Returns:
        area (torch.Tensor): Shape (batch_sz, batch_sz), where smaller is better.
    """
    a2 = torch.sum(anchor * anchor, dim=1, keepdim=True)      # (B, 1)
    b2 = torch.sum(mod_b * mod_b, dim=1).unsqueeze(0)         # (1, B)
    c2 = torch.sum(mod_c * mod_c, dim=1).unsqueeze(0)         # (1, B)

    ab = anchor @ mod_b.t()                                   # (B, B)
    ac = anchor @ mod_c.t()                                   # (B, B)
    # Both non-anchor modalities belong to the same candidate j, so this term
    # must contain <b_j, c_j> and be broadcast over anchor index i.
    bc = torch.sum(mod_b * mod_c, dim=1).unsqueeze(0)           # (1, B)

    u_norm = a2 + b2 - (2.0 * ab)                             # ||a_i - b_j||^2
    v_norm = a2 + c2 - (2.0 * ac)                             # ||a_i - c_j||^2
    uv_dot = a2 - ab - ac + bc                                # (a_i - b_j)^T (a_i - c_j)

    det_term = torch.clamp(u_norm * v_norm - uv_dot * uv_dot, min=0.0)

    if squared:
        return 0.5 * det_term
    else:
        return 0.5 * torch.sqrt(det_term + eps)


def compute_triangle_infonce(anchor, mod_b, mod_c, logit_scale, squared=True, eps=1e-8):
    """
    Symmetric InfoNCE-style TRIANGLE loss for one anchor choice.

    Positives are on the diagonal: anchor[i] should match the paired tuple
    (mod_b[i], mod_c[i]) by minimizing triangle area.

    Args:
        anchor, mod_b, mod_c (torch.Tensor): Shape (batch_sz, d)
        logit_scale (torch.Tensor): Learned temperature parameter
            (same semantics as in CLIP: larger => sharper logits)
        squared (bool): Whether to use squared-area surrogate
        eps (float): Numerical stability epsilon

    Returns:
        loss (torch.Tensor)
    """
    area = compute_triangle_area_matrix_efficient(anchor, mod_b, mod_c, squared=squared, eps=eps)

    # Smaller area = better match, so negate it to obtain logits.
    logits = apply_logit_scale(-area, logit_scale)

    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_forward = F.cross_entropy(logits, labels)
    loss_backward = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_forward + loss_backward)


def triangle(r_a, r_b, r_c, logit_scale, negative_sampling=None, bias=None, **kwargs):
    """
    Computes a plug-and-play TRIANGLE loss for 3 modalities.

    This implementation is made API-compatible with your clip(...) objective.
    Unlike the task-specific official example that uses one anchor modality
    (e.g. language vs paired video+audio), this version is modality-agnostic:
    it averages TRIANGLE InfoNCE over all 3 possible anchor choices.

    Args:
        r_a, r_b, r_c (torch.Tensor): Representation vectors of size (batch_sz, d).
        logit_scale (torch.Tensor): Learned temperature parameter.
        negative_sampling (str): Ignored. Kept for API compatibility.
        bias (torch.Tensor): Ignored. Kept for API compatibility.
        **kwargs:
            squared (bool, optional): If True, use squared-area surrogate
                (default: True, matching the repo's practical implementation).
            eps (float, optional): Numerical stability epsilon for sqrt mode.

    Returns:
        loss (torch.Tensor): Average TRIANGLE loss across the 3 anchor choices.
    """
    squared = kwargs.get("squared", True)
    eps = kwargs.get("eps", 1e-8)

    loss_a = compute_triangle_infonce(r_a, r_b, r_c, logit_scale, squared=squared, eps=eps)
    loss_b = compute_triangle_infonce(r_b, r_a, r_c, logit_scale, squared=squared, eps=eps)
    loss_c = compute_triangle_infonce(r_c, r_a, r_b, logit_scale, squared=squared, eps=eps)

    return (loss_a + loss_b + loss_c) / 3.0
