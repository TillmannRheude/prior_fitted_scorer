# Prior-Fitted and Frozen Scorers for Multimodal Contrastive Learning

[![arXiv](https://img.shields.io/badge/arXiv-xxx-b31b1b.svg)](https://arxiv.org/pdf/xxx)

Multimodal contrastive learning typically relies on manually specified scoring functions, such as cosine similarity, to quantify the similarity between representations from different modalities. Several such functions have been recently proposed to extend contrastive learning beyond two modalities. In principle, however, any function that assigns high compatibility to matched samples and low compatibility to mismatched ones could serve as a scorer. Motivated by this broader view, we instantiate the scorer with a Transformer and find that even a randomly initialized model, when kept frozen, can outperform strong handcrafted multimodal contrastive baselines, whereas optimizing the scorer jointly with the encoders is less effective. Building on this observation, we introduce the Prior-Fitted Scorer (PFS): a Transformer scorer pre-trained on synthetic multimodal data generated from structural causal model (SCM) priors before transferring it to real-world data. The SCM exposes the scorer to diverse dependency structures, including shared, pairwise, and private information, providing a structured prior over higher-order cross-modal similarity. Across four real-world datasets, frozen Transformer scorers are competitive to contrastive baselines, and prior-fitting further improves the random initialization. More broadly, our results suggest that multimodal contrastive learning need not rely on manual similarity functions: Transformers can provide strong inductive biases, while causal pretraining can further shape these biases.

## Citation

Citation information is omitted during anonymous review.
