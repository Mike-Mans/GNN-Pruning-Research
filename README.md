# GNN-Pruning-Research

CS260C Spring 2026 course project — Michael Mansour.

**Core question:** How do topological features, specifically node degree and homophily, influence the efficacy of activation-aware pruning in Graph Neural Networks?

The project extends Wanda-style pruning (Sun et al., ICLR 2024), originally developed for LLMs, to GNNs. In LLMs, weights are scored by `|W_pq| * ||X_q||_2` — weight magnitude times input activation norm. In GNNs, post-aggregation activations depend on graph topology (a hub aggregating 50 neighbors produces far larger activations than a leaf with 2), so the same scoring rule interacts with degree and homophily in ways that don't appear in the LLM setting. The full proposal is in [proposal.tex](proposal.tex).

## Status

Scaffolding only — no implementation yet. The directory tree below is in place; submodules are empty.

## Methods

Five conditions, all evaluated as one-shot post-training pruning (no retraining), at sparsity levels 10%–90%:

1. **No pruning** — dense baseline / accuracy ceiling.
2. **Magnitude pruning** — `S_pq = |W_pq|`.
3. **Wanda-Uniform** — `S_pq = |W_pq| * ||X_q||_2`, activation norm averaged uniformly across nodes.
4. **Wanda-Degree-Weighted** — node contributions weighted by degree (hubs count more).
5. **Wanda-Per-Class** — class-specific activation norms averaged across classes (mitigates majority-class domination, relevant for heterophilic and imbalanced graphs).

Architectures: GCN, GAT, GraphSAGE (2-layer, hidden dim 256). Metrics: accuracy on class-balanced datasets; F1 + recall on the imbalanced Ethereum AML graph.

## Datasets

- **Homophilic:** Cora, Citeseer, Pubmed, Flickr, Arxiv, Yelp, Reddit, BBBP, Proteins.
- **Heterophilic:** Squirrel, Wisconsin.
- **Custom:** Ethereum AML graph (extreme class imbalance, heavy-tailed degree distribution).

## Repository layout

```
GNN-Pruning-Research/
├── src/
│   └── gnn_pruning/         # importable package — all source code lives here
│       ├── data/            # dataset loaders, homophily / degree stats
│       ├── models/          # GCN, GAT, GraphSAGE
│       ├── pruning/         # the 5 scoring variants + sparsity application
│       ├── training/        # training loop, activation-collection forward pass
│       ├── eval/            # accuracy / F1 / recall, sparsity-sweep aggregation
│       ├── configs/         # YAML run configs
│       └── cli.py           # pipeline entry point (orchestrates a sweep)
├── tests/                   # pytest tests for scoring + model sanity
├── data/                    # raw / processed datasets (contents gitignored)
├── docs/
│   └── notebooks/           # exploratory + final-results Jupyter notebooks
├── proposal.tex
├── pyproject.toml
└── README.md
```

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Running experiments

Once the pipeline is implemented, a typical sweep will run as:

```sh
python -m gnn_pruning.cli sweep --config src/gnn_pruning/configs/<name>.yaml
```

A config declares the dataset, architecture, sparsity grid, and pruning variants for one run. Results land under `data/outputs/` (gitignored).

## References

1. M. Sun et al. *A Simple and Effective Pruning Approach for Large Language Models.* ICLR 2024.
2. C. Liu et al. *Comprehensive Graph Gradual Pruning for Sparse Training in GNNs.* IEEE TNNLS 2024.
3. K. Khedri et al. *Pruning and Quantization Impact on Graph Neural Networks.* arXiv:2510.22058, 2025.
4. H. Zhou et al. *Accelerating Large Scale Real-Time GNN Inference using Channel Pruning.* PVLDB 2021.
