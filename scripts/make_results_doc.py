"""Generate results/results_comprehensive.md from the summary.csv files.

Run from the repo root (or set GNN_ROOT). Used both locally and by the Colab
reproduction notebook.
"""
import pandas as pd, numpy as np, os, sys
from collections import Counter
ROOT = os.environ.get("GNN_ROOT", os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src"))
from gnn_pruning.data import DATASET_META, load_dataset
from gnn_pruning.pruning.no_pruning import _infer_dims
from gnn_pruning.training import _flatten_y

_info = {}
def ds_info(d):
    """Load each dataset once; return (num_classes, edge_homophily or None)."""
    if d not in _info:
        meta = DATASET_META[d]
        data = load_dataset(d)
        _, out = _infer_dims(data, meta.task)
        if meta.task == "graph-classification":
            h = None
        else:
            y = _flatten_y(data.y)
            h = None if y.dim() > 1 else float(
                (y[data.edge_index[0]] == y[data.edge_index[1]]).float().mean())
        _info[d] = (out, h)
    return _info[d]

def num_classes(d):
    return ds_info(d)[0]

def homophily(d):
    return ds_info(d)[1]

def hom_label(d):
    base = DATASET_META[d].homophily.capitalize()
    h = homophily(d)
    return f"{base} ({h:.2f})" if h is not None else f"{base} (—)"

import os
ROOT = os.environ.get("GNN_ROOT", os.getcwd())
PM = ["magnitude", "wanda-uniform", "wanda-degree", "wanda-per-class"]
LBL = {"magnitude": "Magnitude", "wanda-uniform": "Wanda-Uniform",
       "wanda-degree": "Wanda-Degree", "wanda-per-class": "Wanda-Per-Class"}
SP = [round(0.1 * i, 1) for i in range(1, 10)]
S = {m: pd.read_csv(f"{ROOT}/results/{m}/summary.csv") for m in PM}
DENSE = pd.read_csv(f"{ROOT}/results/no-pruning/summary.csv").set_index(["dataset", "architecture"])

def val(m, d, a, s):
    r = S[m][(S[m].dataset == d) & (S[m].architecture == a) & (S[m].sparsity == s)]
    return float(r.metric_value.values[0]) if len(r) else None

def metric_name(d, a):
    r = S["magnitude"][(S["magnitude"].dataset == d) & (S["magnitude"].architecture == a)]
    return r.metric_name.values[0]

def dense_val(d, a):
    return float(DENSE.loc[(d, a), "metric_value"])

def nruns(d, a):
    return int(DENSE.loc[(d, a), "n_runs"])

def cell_table(d, a):
    """Method x sparsity table, row-best bolded."""
    out = ["| Sparsity | Magnitude | Wanda-Uniform | Wanda-Degree | Wanda-Per-Class |",
           "|---|---|---|---|---|"]
    for s in SP:
        vs = {m: val(m, d, a, s) for m in PM}
        best = max(v for v in vs.values() if v is not None)
        cells = []
        for m in PM:
            v = vs[m]
            txt = f"{v:.3f}" if v is not None else "—"
            if v is not None and abs(v - best) < 1e-9:
                txt = f"**{txt}**"
            cells.append(txt)
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    return "\n".join(out)

COMPETITIVE_T = 0.95  # a "win" must retain >=95% of the dense baseline (<=5% rel. drop)

def wins(cells, lo=0.1, hi=0.9, thresh=COMPETITIVE_T):
    """Competitive win counts: best of the 4 pruning methods at a cell x sparsity,
    counted only if that best retains >= thresh of the cell's dense baseline.
    Returns (Counter, none_competitive_count)."""
    c = Counter()
    none = 0
    for d, a in cells:
        dv = dense_val(d, a)
        for s in SP:
            if not (lo <= s <= hi):
                continue
            vs = {m: val(m, d, a, s) for m in PM}
            if any(v is None for v in vs.values()):
                continue
            best = max(vs, key=vs.get)
            if vs[best] >= thresh * dv:
                c[best] += 1
            else:
                none += 1
    return c, none

# ---- category definitions ----
HET_GPR = [("cornell", "gprgnn"), ("texas", "gprgnn"), ("wisconsin", "gprgnn"), ("actor", "gprgnn")]
HET_VAN = [("cornell", "gcn"), ("cornell", "gat"), ("texas", "gcn"),
           ("wisconsin", "gcn"), ("actor", "gcn"), ("actor", "gat")]
HOM_SM = [("cora", "gcn"), ("cora", "gat"), ("citeseer", "gcn"), ("citeseer", "gat"),
          ("pubmed", "gcn"), ("pubmed", "gat"), ("cs", "gcn"), ("physics", "gcn"),
          ("photo", "gcn"), ("photo", "gat"), ("computers", "gcn"), ("computers", "gat")]
HOM_LG = [("ogbn-arxiv", "graphsage"), ("flickr", "graphsage"), ("reddit", "gcn"),
          ("reddit", "graphsage"), ("yelp", "graphsage")]
GRAPH = [("bbbp", "gcn"), ("bbbp", "graphsage"), ("proteins", "gcn"), ("proteins", "graphsage")]
ALL = HET_GPR + HET_VAN + HOM_SM + HOM_LG + GRAPH

def winline(cells):
    c, none = wins(cells)
    tot = sum(c.values()) + none
    if not tot:
        return "_no complete cells_"
    body = "  ·  ".join(f"{LBL[m]}: **{c[m]}**" for m in PM)
    return f"{body}  ·  _none-competitive: {none}_  (of {tot} cell×sparsity points)"

def plot_md(d, a):
    rel = f"wanda-per-class/plots/accuracy_vs_sparsity_{d}_{a}.png"
    if os.path.exists(f"{ROOT}/results/{rel}"):
        return f"![{d}/{a} overlay]({rel})"
    return "_(overlay plot unavailable — cell did not complete)_"

def section(cells):
    parts = []
    for d, a in cells:
        mn = {"accuracy": "accuracy", "macro_f1": "macro-F1", "micro_f1": "micro-F1"}[metric_name(d, a)]
        parts.append(f"#### {d} · {a.upper()}\n")
        parts.append(f"*Metric: {mn} · dense baseline (0% sparsity): "
                     f"**{dense_val(d, a):.3f}** · averaged over {nruns(d, a)} "
                     f"run{'s' if nruns(d, a) > 1 else ''}*\n")
        parts.append(cell_table(d, a) + "\n")
        parts.append(plot_md(d, a) + "\n")
    return "\n".join(parts)

# ---- assemble ----
md = []
md.append("# Comprehensive Results — GNN Activation-Based Pruning\n")
md.append("Generated from `results/<method>/summary.csv`. Five methods compared across "
          "every completed (dataset, architecture) cell at 9 sparsity levels (0.1–0.9). "
          "Each cell shows the per-sparsity table (best method per row **bold**) and the "
          "overlaid accuracy-vs-sparsity plot from `results/wanda-per-class/plots/`, which "
          "draws all four pruning methods plus the dense reference.\n")

md.append("## Methods\n")
md.append("- **No-Pruning** — dense baseline (0% sparsity).\n"
          "- **Magnitude** — prune lowest `|W|` per layer (CGP-style baseline).\n"
          "- **Wanda-Uniform** — prune lowest `|W|·‖X‖₂` (activation-aware).\n"
          "- **Wanda-Degree** — Wanda with degree-weighted activations (`√deg·X`).\n"
          "- **Wanda-Per-Class** — Wanda with class-balanced activation norms (the proposal's hypothesis).\n")

md.append("## Metric note\n")
md.append("Heterophilic datasets (Cornell/Texas/Wisconsin/Actor) report **macro-F1**, the "
          "metric aligned with the class-imbalance hypothesis; these are 5-class, imbalanced "
          "graphs, so macro-F1 sits well below accuracy (e.g. Wisconsin/GPR-GNN ≈ 0.59 macro-F1 "
          "≈ 0.77 accuracy). Multi-label Yelp reports **micro-F1**; everything else reports "
          "**accuracy**. All numbers are single-seed except the heterophilic cells, which are "
          "averaged over the 10 Geom-GCN splits.\n")

md.append("## Coverage\n")
md.append("31 of 33 (dataset, architecture) cells completed across all 5 methods. "
          "**Not shown:** `reddit/gat` (full-batch attention OOM, ~178 GiB — infeasible on any "
          "GPU) and `ogbn-products/graphsage` (OGB interactive-download prompt in the headless "
          "run — recoverable). Reddit is covered by GCN and GraphSAGE.\n")

md.append("## Headline finding — competitive win counts\n")
md.append("For each cell × sparsity level (0.1–0.9), we tally which of the **four pruning methods** "
          "is best — **but a win only counts if that method stays competitive with the dense model**, "
          "i.e. retains ≥ 95% of the cell's dense-baseline metric (≤ 5% relative drop). A win among "
          "collapsed models is not a win. **None-competitive** counts cell × sparsity points where "
          "*even the best* pruning method fell below that bar (the over-pruned regime). The dense "
          "baseline itself is the reference, not a competitor (it only exists at 0% sparsity).\n")
md.append("| Group | Magnitude | Wanda-Uniform | Wanda-Degree | Wanda-Per-Class | None-competitive |\n|---|---|---|---|---|---|")
for name, cells in [("**All cells**", ALL), ("Heterophilic (GPR-GNN)", HET_GPR),
                    ("Heterophilic (vanilla GCN/GAT)", HET_VAN),
                    ("Homophilic small/medium", HOM_SM), ("Homophilic large", HOM_LG),
                    ("Graph classification", GRAPH)]:
    c, none = wins(cells)
    row = " | ".join(str(c[m]) for m in PM)
    md.append(f"| {name} | {row} | {none} |")
md.append("")
md.append("> **Wanda-Per-Class — the proposal's hypothesized method — is never the best in any group, "
          "and ranks last (or tied-last) on every node-classification group.** It provides no benefit "
          "even in the heterophilic / class-imbalanced regime it was designed for, now tested on a "
          "functional GPR-GNN backbone. Which of the *other* three leads varies by group and is "
          "sensitive to the threshold (Magnitude on homophilic and all-cells; Wanda-Uniform on "
          "graph-classification and vanilla-heterophilic; the margins are small) — but the **bottom** "
          "of the ranking is stable: the class-aware refinement doesn't earn its complexity. The "
          "per-class result is robust to the competitiveness bar — last at 90%, 95%, and 99% of dense "
          "alike (only the none-competitive total shifts: 12 / 24 / 55 of 279 across all cells).\n")
md.append("_Note: \"best\" is rank at matched sparsity and says nothing about margin — at low sparsity "
          "all four methods sit within ~0.3 points, so the per-cell tables below show the actual "
          "magnitudes. Per-cell tables bold the row-max (raw best) regardless of competitiveness._\n")

md.append("## Dense baselines (0% sparsity), all cells\n")
md.append("Homophilic datasets first (alphabetical), then heterophilic (alphabetical).\n")
md.append("| Dataset | Homophily | Classes | Architecture | Metric | Dense value | Runs |\n|---|---|---|---|---|---|---|")
# homophilic group (0) before heterophilic group (1); then alphabetical by dataset, arch.
dense_order = sorted(ALL, key=lambda c: (0 if DATASET_META[c[0]].homophily == "homophilic" else 1, c[0], c[1]))
for d, a in dense_order:
    mn = {"accuracy": "accuracy", "macro_f1": "macro-F1", "micro_f1": "micro-F1"}[metric_name(d, a)]
    cls = f"{num_classes(d)}" + ("\\*" if d == "yelp" else "")
    md.append(f"| {d} | {hom_label(d)} | {cls} | {a.upper()} | {mn} | {dense_val(d, a):.3f} | {nruns(d, a)} |")
md.append("")
md.append("\\* Yelp is multi-label (100 binary labels); all other datasets are single-label. "
          "Homophily = edge homophily (fraction of edges joining same-label nodes); "
          "`—` where undefined (graph-classification, multi-label).\n")

# ---- best-result-per-cell coverage matrix (same dataset order as above) ----
SHORT = {"magnitude": "Mag", "wanda-uniform": "Unif", "wanda-degree": "Deg", "wanda-per-class": "PerCls"}
done_cells = set(ALL)
def best_cell(d, a):
    if (d, a) not in done_cells:
        return "—"
    cands = [(dense_val(d, a), "dense")]
    for m in PM:
        for s in SP:
            v = val(m, d, a, s)
            if v is not None:
                cands.append((v, f"{SHORT[m]} @{int(round(s * 100))}%"))
    v, reg = max(cands, key=lambda x: x[0])
    return f"{v:.3f} ({reg})"
seen = set(); ds_order = []
for d, a in dense_order:
    if d not in seen:
        seen.add(d); ds_order.append(d)
md.append("## Best result per (dataset × architecture)\n")
md.append("Each cell is the **best metric observed** for that (dataset × architecture) and the regime "
          "that produced it: `dense` = no pruning, otherwise `Method @ sparsity`. Same dataset "
          "ordering and homophily labels as the table above. **Caveat:** this is the maximum over 37 "
          "configurations per cell (dense + 4 methods × 9 sparsities), so it is an optimistic, "
          "selection-biased estimate — the per-cell tables further down give the full sparsity "
          "curves. `—` = not evaluated (`reddit/gat` OOM; `ogbn-products` download-blocked).\n")
md.append("| Dataset | Homophily | GCN | GAT | GraphSAGE | GPR-GNN |\n|---|---|---|---|---|---|")
for d in ds_order:
    cells = [best_cell(d, a) for a in ["gcn", "gat", "graphsage", "gprgnn"]]
    md.append(f"| {d} | {hom_label(d)} | " + " | ".join(cells) + " |")
md.append("")

md.append("---\n")
md.append("## 1. Heterophilic — GPR-GNN backbone *(primary heterophilic result)*\n")
md.append("The proposal's hypothesis lives here: class-imbalanced, heterophilic graphs. Vanilla "
          "GCN/GAT cannot learn these (Section 2), so we re-ran them with **GPR-GNN**, a "
          "heterophily-capable backbone whose prunable weights are still plain Linears. GPR-GNN "
          "roughly doubles vanilla GCN's macro-F1 (e.g. Wisconsin 0.59 vs 0.31), so the pruning "
          "comparison is finally interpretable — and Per-Class still does not win.\n")
md.append(f"_Win counts here:_ {winline(HET_GPR)}\n")
md.append(section(HET_GPR))

md.append("---\n")
md.append("## 2. Heterophilic — vanilla GCN/GAT *(base model fails to learn — uninterpretable)*\n")
md.append("Included for completeness. On these the dense models barely clear trivial baselines "
          "(macro-F1 0.23–0.31; near or below majority-class accuracy), so method differences are "
          "noise. This is the known heterophily failure of homophily-assuming aggregation, and the "
          "reason Section 1 re-runs them with GPR-GNN.\n")
md.append(f"_Win counts here:_ {winline(HET_VAN)}\n")
md.append(section(HET_VAN))

md.append("---\n")
md.append("## 3. Homophilic — small/medium node classification *(healthy base models)*\n")
md.append("Base models are strong (accuracy 0.68–0.97, far above trivial), so this comparison is "
          "meaningful. Methods are nearly tied until ~70% sparsity; at extreme sparsity "
          "Wanda-Uniform is modestly best and Per-Class is among the weakest.\n")
md.append(f"_Win counts here:_ {winline(HOM_SM)}\n")
md.append(section(HOM_SM))

md.append("---\n")
md.append("## 4. Homophilic — large-scale node classification\n")
md.append("Reddit/Yelp/Arxiv/Flickr. Reddit (GCN/SAGE) and Yelp ran via the sparse-adjacency "
          "(SpMM) path. Baselines are healthy in rank terms (well above majority) but low in "
          "absolute terms vs literature — the 2-layer/100-epoch config is undertuned on these "
          "large graphs, and **Flickr is near-trivial** (+0.02 over majority), so treat Flickr as "
          "uninterpretable.\n")
md.append(f"_Win counts here:_ {winline(HOM_LG)}\n")
md.append(section(HOM_LG))

md.append("---\n")
md.append("## 5. Graph classification\n")
md.append("BBBP and PROTEINS (global-mean-pool readout). Single random 80/10/10 split.\n")
md.append(f"_Win counts here:_ {winline(GRAPH)}\n")
md.append(section(GRAPH))

out = f"{ROOT}/results/results_comprehensive.md"
open(out, "w").write("\n".join(md))
print("wrote", out)
print("lines:", len("\n".join(md).splitlines()), "| cells:", len(ALL))
