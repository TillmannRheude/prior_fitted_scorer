import torch
import torch.nn.functional as F
from losses.utils import apply_logit_scale

def compute_pairwise_clip(x, y, logit_scale):
    """
    Computes pairwise CLIP loss (InfoNCE) between two modalities x and y.
    """
    # x, y: (batch_sz, d) - assumed normalized if embedding_norm=True
    # logits: (batch_sz, batch_sz)
    logits = apply_logit_scale(x @ y.t(), logit_scale)
    
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Symmetric loss: (L_xy + L_yx) / 2
    loss_x = F.cross_entropy(logits, labels)
    loss_y = F.cross_entropy(logits.t(), labels)
    
    return (loss_x + loss_y) / 2.0

def clip(r_a, r_b, r_c, logit_scale, negative_sampling=None, bias=None, **kwargs):
    """
    Computes the CLIP loss for 3 modalities by averaging the pairwise CLIP losses.
    
    Args:
        r_a, r_b, r_c (torch.Tensor): Representation vectors of size (batch_sz, d).
        logit_scale (torch.Tensor): Learned temperature parameter.
        negative_sampling (str): Ignored. Kept for API compatibility.
        bias (torch.Tensor): Ignored. Kept for API compatibility.
        **kwargs: Additional arguments for API compatibility.
        
    Returns:
        loss (torch.Tensor): Average pairwise CLIP loss.
    """
    loss_ab = compute_pairwise_clip(r_a, r_b, logit_scale)
    loss_ac = compute_pairwise_clip(r_a, r_c, logit_scale)
    loss_bc = compute_pairwise_clip(r_b, r_c, logit_scale)
    
    return (loss_ab + loss_ac + loss_bc) / 3.0


def bimodal_clip(r_a, r_b, logit_scale, negative_sampling=None, bias=None, **kwargs):
    """
    Computes CLIP loss for 2 modalities.

    Args:
        r_a, r_b (torch.Tensor): Representation vectors of size (batch_sz, d).
        logit_scale (torch.Tensor): Learned temperature parameter.
        negative_sampling (str): Ignored. Kept for API compatibility.
        bias (torch.Tensor): Ignored. Kept for API compatibility.
        **kwargs: Additional arguments for API compatibility.

    Returns:
        loss (torch.Tensor): Symmetric pairwise CLIP loss.
    """
    return compute_pairwise_clip(r_a, r_b, logit_scale)
