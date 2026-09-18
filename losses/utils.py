def scale_mip_dvs(score_matrix, d, M):
    """
    Scale raw multi-way MIP scores using generalized Deterministic Variance Scaling.
    scale_base = d ** ((M - 1) / 2)  (valid for any M >= 2)
    """
    scale_base = d ** ((M - 1) / 2)
    return score_matrix * scale_base


def apply_logit_scale(logits, logit_scale):
    """
    Multiply logits by `logit_scale` when provided; otherwise leave them unchanged.
    """
    if logit_scale is None:
        return logits
    return logit_scale * logits
