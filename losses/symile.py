import torch
import torch.nn.functional as F
from losses.utils import apply_logit_scale, scale_mip_dvs

def compute_logits_neg_sampling_n(x, y, z):
    # shuffle rows of y and z
    y_shuff = y[torch.randperm(y.shape[0])]
    z_shuff = z[torch.randperm(z.shape[0])]
    logits_x = x @ torch.t(y_shuff * z_shuff) # (batch_sz, batch_sz)
    MIP_of_pos_triples = (x * y * z).sum(axis=1) # (batch_sz)
    # insert positive triples along diagonal of shuffled logits
    return torch.where(torch.eye(n=x.shape[0]).to(x.device) > 0.5, MIP_of_pos_triples, logits_x)

def compute_logits_neg_sampling_n_squared(x, y, z):
    y_z = []
    for i in range(y.shape[0]):
        y_z.append(y * z)
        z = torch.roll(z, shifts=1, dims=0)

    # concatenate elements in y_z so that y_z has shape (n^2, d) where each row
    # is a different element-wise product of a row from y and a row from z
    y_z = torch.cat(y_z, 0)

    # return logits with shape (n, n^2) where each row is the multilinear inner
    # product between that row in x and each row from y_z
    logits = x @ y_z.T
    return logits

def symile(
    r_a, 
    r_b, 
    r_c, 
    logit_scale, 
    negative_sampling, 
    bias=None, 
    labels=None, 
    candidates=None,
    pair_num_negatives=None,
):
    """
    Computes the Symile loss for a batch of representations. The final Symile
    loss is an average of the loss terms where each modality is treated as the
    anchor in turn.

    Returns:
        (torch.Tensor): Average over the losses where each modality is treated
                        as the anchor in turn.
    """
    d = r_a.shape[-1]
    M = 3

    if candidates is None:
        c_a, c_b, c_c = r_a, r_b, r_c
    else:
        c_a, c_b, c_c = candidates
    cand_pools = [c_a, c_b, c_c]
    emb_local = [r_a, r_b, r_c]

    if negative_sampling == "pair":
        B = r_a.shape[0]
        if labels is None:
            # local in-batch positives
            labels = torch.arange(B, device=r_a.device)

        def _loss_for_target(t: int) -> torch.Tensor:
            c_t = cand_pools[t]  # (N, d)
            N = c_t.shape[0]
            if N < 2:
                return (r_a.sum() * 0.0) + 0.0

            K = int(pair_num_negatives)
            K = max(1, min(K, N - 1))

            pos = labels.to(device=r_a.device, dtype=torch.long)  # (B,)
            if pos.min().item() < 0 or pos.max().item() >= N:
                raise ValueError(
                    f"Labels out of range for target {t}: got [{int(pos.min())},{int(pos.max())}] but N={N}"
                )

            # sample negatives uniformly (cheap; no strict w/o-replacement)
            neg = torch.randint(0, N - 1, (B, K), device=r_a.device, dtype=torch.long)  # [0, N-2]
            neg = neg + (neg >= pos[:, None]).long()  # shift to skip pos -> now in [0,N-1]\{pos}

            idx = torch.cat([pos[:, None], neg], dim=1)  # (B, K+1), positive in col 0
            KK = K + 1

            # gather target candidates: (B, K+1, d)
            cand_t = c_t[idx]  # (B, KK, d)

            # broadcast the other (local) modalities across the KK candidates
            prod = torch.ones((B, KK, d), device=r_a.device, dtype=r_a.dtype)
            for m in range(3):
                if m == t:
                    continue
                x0 = emb_local[m]  # (B, d)
                prod = prod * x0[:, None, :].expand(B, KK, d)

            raw = (prod * cand_t).sum(dim=2)  # (B, KK)

            # DVS scaling + temperature
            raw = scale_mip_dvs(raw, d=d, M=M)
            logits = apply_logit_scale(raw, logit_scale)
            if bias is not None:
                logits = logits + bias  # no-op for symile, but harmless

            y = torch.zeros((B,), device=r_a.device, dtype=torch.long)
            return F.cross_entropy(logits, y)

        loss_a = _loss_for_target(0)
        loss_b = _loss_for_target(1)
        loss_c = _loss_for_target(2)
        return (loss_a + loss_b + loss_c) / 3.0

    if negative_sampling == "n":
        raw_a = compute_logits_neg_sampling_n(r_a, r_b, r_c)
        raw_b = compute_logits_neg_sampling_n(r_b, r_a, r_c)
        raw_c = compute_logits_neg_sampling_n(r_c, r_a, r_b)
    elif negative_sampling == "n_squared":
        raw_a = compute_logits_neg_sampling_n_squared(r_a, c_b, c_c)
        raw_b = compute_logits_neg_sampling_n_squared(r_b, c_a, c_c)
        raw_c = compute_logits_neg_sampling_n_squared(r_c, c_a, c_b)
    else:
        raise ValueError("negative_sampling must be either 'n' or 'n_squared'.")
    
    # DVS scaling
    raw_a = scale_mip_dvs(raw_a, d, M)
    raw_b = scale_mip_dvs(raw_b, d, M)
    raw_c = scale_mip_dvs(raw_c, d, M)
    # Temperature
    logits_a = apply_logit_scale(raw_a, logit_scale)
    logits_b = apply_logit_scale(raw_b, logit_scale)
    logits_c = apply_logit_scale(raw_c, logit_scale)

    if labels is None:
        labels = torch.arange(logits_a.shape[0], device=r_a.device)

    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    loss_c = F.cross_entropy(logits_c, labels)
    return (loss_a + loss_b + loss_c) / 3.0
