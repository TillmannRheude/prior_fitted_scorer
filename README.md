# Prior-Fitted and Frozen Scorers for Multimodal Contrastive Learning

[![arXiv](https://img.shields.io/badge/arXiv-xxx-b31b1b.svg)](https://arxiv.org/pdf/xxx)

Multimodal contrastive learning typically relies on manually specified scoring functions, such as cosine similarity, to quantify the similarity between representations from different modalities. Several such functions have been recently proposed to extend contrastive learning beyond two modalities. In principle, however, any function that assigns high compatibility to matched samples and low compatibility to mismatched ones could serve as a scorer. Motivated by this broader view, we instantiate the scorer with a Transformer and find that even a randomly initialized model, when kept frozen, can outperform strong handcrafted multimodal contrastive baselines, whereas optimizing the scorer jointly with the encoders is less effective. Building on this observation, we introduce the Prior-Fitted Scorer (PFS): a Transformer scorer pre-trained on synthetic multimodal data generated from structural causal model (SCM) priors before transferring it to real-world data. The SCM exposes the scorer to diverse dependency structures, including shared, pairwise, and private information, providing a structured prior over higher-order cross-modal similarity. Across four real-world datasets, frozen Transformer scorers are competitive to contrastive baselines, and prior-fitting further improves the random initialization. More broadly, our results suggest that multimodal contrastive learning need not rely on manual similarity functions: Transformers can provide strong inductive biases, while causal pretraining can further shape these biases.

## Using a frozen Transformer scorer

The scorer consumes one embedding per modality and returns one scalar compatibility
score per tuple. The following creates the small, randomly initialized baseline. All
linear and attention projections use Xavier-uniform initialization, as implemented in
[`TransformerScorer`](architecture.py).

```python
import torch
import torch.nn.functional as F
from architecture import TransformerScorer

embedding_dim = 128
scorer = TransformerScorer(
    transformer_params={
        "d_model": 256,
        "nhead": 2,
        "num_layers": 2,
        "dropout": 0.0,
        "disable_flash_mem_efficient_sdp": False,
        "initialization": "xavier",
    },
    input_dim=embedding_dim,
    max_modalities=3,
    proj_output_dim=1,
)
scorer.requires_grad_(False)
```

For pair sampling, each item in a batch is the positive candidate for the other
modalities. We replace one target modality at a time with `K` randomly sampled
non-matching items. Thus, column zero is always the positive and the remaining
columns are negatives:

```python
def pair_sampled_loss(scorer, embeddings, num_negatives=128):
    """
    Pair-sampled loss for three tensors of shape (batch, embedding_dim).
    """
    embeddings = [F.normalize(x, dim=-1) for x in embeddings]
    batch_size = embeddings[0].shape[0]
    k = min(num_negatives, batch_size - 1)
    positive = torch.arange(batch_size, device=embeddings[0].device)
    losses = []

    for target in range(3):
        # Sample from [0, batch_size) while excluding each row's positive index.
        negative = torch.randint(
            batch_size - 1, (batch_size, k), device=positive.device
        )
        negative += (negative >= positive[:, None]).long()
        candidate_idx = torch.cat([positive[:, None], negative], dim=1)

        candidate_tuples = []
        for modality, x in enumerate(embeddings):
            if modality == target:
                paired = x[candidate_idx.reshape(-1)]
            else:
                paired = (
                    x[:, None, :]
                    .expand(batch_size, k + 1, -1)
                    .reshape(batch_size * (k + 1), -1)
                )
            candidate_tuples.append(paired)

        logits = scorer(candidate_tuples).squeeze(-1).view(batch_size, k + 1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits, labels))

    return torch.stack(losses).mean()

# Outputs from three trainable encoders, here dummies
embeddings = [
    torch.randn(32, embedding_dim, requires_grad=True) for _ in range(3)
]
loss = pair_sampled_loss(scorer, embeddings, num_negatives=16)
loss.backward()  # gradients pass through the frozen scorer into the encoders
```

To use the prior-fitted scorer (PFS) without Lightning, construct the architecture
used for SCM pretraining and load the tensor-only state dictionary:

```python
pfs = TransformerScorer(
    transformer_params={
        "d_model": 512,
        "nhead": 2,
        "num_layers": 5,
        "dropout": 0.0
    },
    input_dim=128,
    max_modalities=3,
    proj_output_dim=1,
)
state_dict = torch.load(
    "pfs.pt",
    weights_only=True,
)
pfs.load_state_dict(state_dict, strict=True)
pfs.requires_grad_(False)
```

## Citation
If you find this work useful, please cite:
```bibtex
@unpublished{Rheude_Eils_Wild_2026,
  title={Prior-Fitted and Frozen Scorers for Multimodal Contrastive Learning},
  author={Tillmann Rheude and Roland Eils and Benjamin Wild},
  year={2026},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
