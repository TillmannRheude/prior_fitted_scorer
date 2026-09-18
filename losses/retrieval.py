import itertools

import torch

from losses.gram import compute_gramian_volume_matrix
from losses.utils import apply_logit_scale, scale_mip_dvs


def zeroshot_retrieval_logits(
    r_x: torch.Tensor,
    rep_list: list[torch.Tensor],
    logit_scale_exp: torch.Tensor,
    bias: torch.Tensor = None,
    modelname: str = "symile",
):
    if modelname == "symile":
        product = torch.ones_like(rep_list[0])
        for r in rep_list:
            product *= r

        logits = product @ torch.t(r_x)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        d = r_x.shape[-1]
        m = len(rep_list) + 1
        logits = scale_mip_dvs(logits, d, m)

    elif modelname == "clip":
        rep_list = [rep.unsqueeze(0) if rep.dim() == 1 else rep for rep in rep_list]

        logits = torch.zeros_like(rep_list[0] @ torch.t(r_x))
        for rep in rep_list:
            logits = logits + (rep @ torch.t(r_x))

        context_pair_scores = torch.zeros((rep_list[0].shape[0], 1), device=rep_list[0].device)
        for x, y in itertools.combinations(rep_list, 2):
            context_pair_scores = context_pair_scores + torch.diagonal(x @ torch.t(y)).unsqueeze(1)

        logits = logits + context_pair_scores

    elif modelname == "triangle":
        if len(rep_list) != 2:
            raise ValueError("For modelname='triangle', rep_list must contain exactly two query tensors.")

        cand_rep = r_x.unsqueeze(0) if r_x.dim() == 1 else r_x
        query_b = rep_list[0].unsqueeze(0) if rep_list[0].dim() == 1 else rep_list[0]
        query_c = rep_list[1].unsqueeze(0) if rep_list[1].dim() == 1 else rep_list[1]

        cand_norm2 = torch.sum(cand_rep * cand_rep, dim=1).unsqueeze(0)
        query_b_norm2 = torch.sum(query_b * query_b, dim=1, keepdim=True)
        query_c_norm2 = torch.sum(query_c * query_c, dim=1, keepdim=True)

        b_cand = query_b @ cand_rep.t()
        c_cand = query_c @ cand_rep.t()
        b_c = torch.sum(query_b * query_c, dim=1, keepdim=True)

        u_norm = cand_norm2 + query_b_norm2 - (2.0 * b_cand)
        v_norm = cand_norm2 + query_c_norm2 - (2.0 * c_cand)
        uv_dot = cand_norm2 - b_cand - c_cand + b_c

        det_term = torch.clamp(u_norm * v_norm - uv_dot * uv_dot, min=0.0)
        logits = -(0.5 * det_term)

    elif modelname == "gram":
        if len(rep_list) != 2:
            raise ValueError("For modelname='gram', rep_list must contain exactly two query tensors.")

        cand_rep = r_x.unsqueeze(0) if r_x.dim() == 1 else r_x
        query_1 = rep_list[0].unsqueeze(0) if rep_list[0].dim() == 1 else rep_list[0]
        query_2 = rep_list[1].unsqueeze(0) if rep_list[1].dim() == 1 else rep_list[1]

        logits = -compute_gramian_volume_matrix(cand_rep, query_1, query_2).t()

    else:
        raise ValueError(f"Unsupported retrieval modelname={modelname!r}.")

    if logits.dim() != 2:
        raise ValueError(f"Retrieval logits must be 2D, got {tuple(logits.shape)}.")

    logits = apply_logit_scale(logits, logit_scale_exp)
    if bias is not None:
        logits = logits + bias
    return logits
