import torch
import torch.nn.functional as F
from losses.utils import apply_logit_scale


def compute_gramian_volume_matrix(anchor, *inputs, eps=1e-8):
    """
    Computes the pairwise Gramian volume matrix between an anchor batch and the
    paired tuples formed by the modalities in `inputs`.

    Args:
        anchor (torch.Tensor): Shape (B1, D).
        *inputs (torch.Tensor): Each tensor has shape (B2, D). Row j across all
            inputs is treated as one paired candidate tuple.
        eps (float): Numerical stability constant.

    Returns:
        torch.Tensor: Shape (B1, B2), where entry (i, j) is the parallelotope
            volume formed by anchor[i] together with inputs[0][j], ... inputs[M][j].
    """
    if len(inputs) == 0:
        raise ValueError("compute_gramian_volume_matrix requires at least one input modality.")

    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]

    if any(x.shape[0] != batch_size2 for x in inputs):
        raise ValueError("All input modalities must share the same batch size.")

    aa = torch.einsum("bi,bi->b", anchor, anchor).unsqueeze(1).expand(-1, batch_size2)
    a_inputs = [anchor @ x.t() for x in inputs]

    input_dot_products = []
    for input_i in inputs:
        row = []
        for input_j in inputs:
            dot_ij = torch.einsum("bi,bi->b", input_i, input_j).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_ij)
        input_dot_products.append(row)

    gram = torch.stack(
        [
            torch.stack([aa] + a_inputs, dim=-1),
            *[
                torch.stack([a_inputs[i]] + input_dot_products[i], dim=-1)
                for i in range(len(inputs))
            ],
        ],
        dim=-2,
    )

    gram_det = torch.det(gram.float())
    return torch.sqrt(torch.clamp(gram_det, min=eps)).to(anchor.dtype)


def compute_gram_infonce(anchor, *inputs, logit_scale, eps=1e-8):
    """
    Symmetric InfoNCE-style GRAM loss for one anchor choice.
    """
    volume = compute_gramian_volume_matrix(anchor, *inputs, eps=eps)
    logits = apply_logit_scale(-volume, logit_scale)

    labels = torch.arange(logits.shape[0], device=logits.device)

    loss_forward = F.cross_entropy(logits, labels)
    loss_backward = F.cross_entropy(logits.t(), labels)

    return 0.5 * (loss_forward + loss_backward)


def gram(r_a, r_b, r_c, logit_scale, negative_sampling=None, bias=None, **kwargs):
    """
    Plug-and-play 3-modal GRAM objective.

    Like `clip(...)` and `triangle(...)` in this codebase, this wrapper ignores
    Symile-specific negative-sampling options and averages the loss over the
    three possible anchor choices.
    """
    eps = kwargs.get("eps", 1e-8)

    loss_a = compute_gram_infonce(r_a, r_b, r_c, logit_scale=logit_scale, eps=eps)
    loss_b = compute_gram_infonce(r_b, r_a, r_c, logit_scale=logit_scale, eps=eps)
    loss_c = compute_gram_infonce(r_c, r_a, r_b, logit_scale=logit_scale, eps=eps)

    return (loss_a + loss_b + loss_c) / 3.0
