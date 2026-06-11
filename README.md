# GNN-Pruning-Research

CS260C Spring 2026 course project — Michael Mansour.

**Core question:** How do topological features, specifically node degree and homophily, influence the efficacy of activation-aware pruning in Graph Neural Networks?

The project extends Wanda-style pruning (Sun et al., ICLR 2024), originally developed for LLMs, to GNNs. In LLMs, weights are scored by `|W_pq| * ||X_q||_2` (weight magnitude times input activation norm). In GNNs, post-aggregation activations depend on graph topology (a hub aggregating 50 neighbors produces far larger activations than a leaf with 2), so the same scoring rule interacts with degree and homophily in ways that don't appear in the LLM setting. The original proposal is in [proposal.tex](proposal.tex).

## Status

Complete. All experiments have run; the full analysis lives in [results/results_comprehensive.md](results/results_comprehensive.md), with per-cell overlay plots in `results/plots/`.

**Headline finding:** the topology-aware refinements do not help. By competitive win count, plain magnitude pruning is the strongest criterion overall, Wanda-Uniform is the best activation-aware variant, and Wanda-Per-Class (the proposal's hypothesized method) ranks last, including in the heterophilic, class-imbalanced regime it was designed for. All methods stay within roughly one point of the dense baseline up to about 70% sparsity.

## Methods

Five conditions, all evaluated as one-shot post-training pruning (no retraining), at sparsity levels 10%–90%:

1. **No pruning** — trains the dense baseline / accuracy ceiling and saves checkpoints.
2. **Magnitude pruning** — `S_pq = |W_pq|`.
3. **Wanda-Uniform** — `S_pq = |W_pq| * ||X_q||_2`, activation norm computed uniformly across calibration nodes.
4. **Wanda-Degree-Weighted** — node contributions weighted by degree (hubs count more).
5. **Wanda-Per-Class** — class-specific activation norms averaged with equal weight across classes (mitigates majority-class domination).

Architectures: GCN, GAT, GraphSAGE (2-layer, hidden dim 256), plus GPR-GNN (hidden dim 256, K = 10) for the heterophilic graphs, where vanilla GCN/GAT fail to learn. Metrics: accuracy on class-balanced datasets, macro-F1 on the heterophilic datasets, micro-F1 on multi-label Yelp.

## Datasets

17 datasets, 31 of 33 (dataset, architecture) cells completed (`reddit/gat` is infeasible due to full-batch attention memory; `ogbn-products` was download-blocked):

- **Homophilic:** Cora, Citeseer, Pubmed, Computers, Photo, CS, Physics, Flickr, ogbn-arxiv, Reddit, Yelp.
- **Heterophilic:** Actor, Cornell, Texas, Wisconsin (10 Geom-GCN splits each).
- **Graph classification:** BBBP, PROTEINS.

## Repository layout

```
GNN-Pruning-Research/
├── src/
│   └── gnn_pruning/         # importable package — all source code lives here
│       ├── data/            # dataset loaders, homophily / degree stats
│       ├── models/          # GCN, GAT, GraphSAGE, GPR-GNN
│       ├── pruning/         # the 5 method pipelines (scoring + sparsity application)
│       ├── training/        # training loop, activation-collection hooks
│       ├── eval/            # accuracy / F1, sparsity-sweep aggregation
│       ├── configs/         # YAML run configs (one per method)
│       └── cli.py           # entry point: `run-cell` (one cell) and `sweep` (full grid)
├── scripts/                 # one launcher per method + results-doc generator + Colab notebook
├── results/                 # summary.csv, metrics.json, run.log per method; plots/; results_comprehensive.md
├── tests/
├── data/                    # raw / processed datasets (contents gitignored)
├── docs/notebooks/          # exploratory + results-analysis Jupyter notebooks
├── proposal.tex
├── pyproject.toml
└── README.md
```

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Requires Python 3.11/3.12, PyTorch, and PyTorch Geometric (installed via `pyproject.toml`). Datasets download automatically on first use into `data/`.

## Reproducing the experiments

Run the dense baseline first (it trains and saves the checkpoints every pruning method loads), then the four pruning sweeps, then regenerate the results document:

```sh
./scripts/run_no_pruning.sh
./scripts/run_magnitude.sh
./scripts/run_wanda_uniform.sh
./scripts/run_wanda_degree.sh
./scripts/run_wanda_per_class.sh
python scripts/make_results_doc.py   # rebuilds results/results_comprehensive.md + plots
```

Each script wraps `python -m gnn_pruning.cli sweep --method <m> --config src/gnn_pruning/configs/<m>.yaml`. The sweep launches one subprocess per (dataset, architecture, seed, split) cell, skips cells whose outputs already exist (use `--force` to re-run), and accepts `--datasets` / `--architectures` filters for a partial run, e.g.:

```sh
./scripts/run_wanda_uniform.sh --datasets cora,wisconsin
```

Per-cell metrics land in `results/<method>/<dataset>/<arch>/seed-*/split-*/metrics.json`, aggregated into `results/<method>/summary.csv`. A from-scratch cloud reproduction notebook is in [scripts/colab_run.ipynb](scripts/colab_run.ipynb).

## References

1. M. Sun et al. *A Simple and Effective Pruning Approach for Large Language Models.* ICLR 2024.
2. C. Liu et al. *Comprehensive Graph Gradual Pruning for Sparse Training in GNNs.* IEEE TNNLS, vol. 35, no. 10, 2024.
3. K. Khedri et al. *Pruning and Quantization Impact on Graph Neural Networks.* arXiv:2510.22058, 2025.
4. H. Zhou et al. *Accelerating Large Scale Real-Time GNN Inference using Channel Pruning.* PVLDB 2021.
5. E. Chien et al. *Adaptive Universal Generalized PageRank Graph Neural Network.* ICLR 2021.
