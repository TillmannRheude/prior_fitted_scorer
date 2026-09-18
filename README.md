# Prior Fitted and Frozen Scorers

[![arXiv](https://img.shields.io/badge/arXiv-xxx-b31b1b.svg)](https://arxiv.org/pdf/xxx)

## Abstract
<div align="justify">
Contrastive learning is increasingly moving toward settings with three or more modalities instead of image-text pairs. Yet, extending models from pairwise to higher-order multimodal alignment can introduce optimization and representation challenges. We identify encoder Jacobian conditioning as a key factor in trimodal contrastive learning: poorly conditioned encoders exhibit collapsing or amplified singular-value spectra, leading to exploding Jacobian condition numbers and degraded multimodal alignment. We introduce geometry-preserving encoders (GPEs) by directly conditioning the Jacobian through regularization and demonstrating that simple modifications like LeakyReLU activations and residual paths recover these geometric benefits. Across a synthetic benchmark and four real-world datasets including missing modalities, improving Jacobian conditioning boosts retrieval and linear probe performance across multiple contrastive objectives, whereas expressive objectives yield little benefit in linear probes. More broadly, our results show that multimodal contrastive learning depends not only on objective expressivity, but also on the geometric and optimization properties of the underlying encoders.
</div>

## Implement a PFS or Frozen Scorer

## Citation
In case you find our work helpful, we're happy if you cite us as following
```bibtex
@unpublished{Rheude_Hegselmann_Eils_Wild_2025, 
      title={Prior-Fitted and Frozen Scorers for Multimodal Contrastive Learning}, 
      author={Tillmann Rheude and Roland Eils and Benjamin Wild},
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={}, 
}