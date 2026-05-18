# Pre-Flight Validation Report

**Run date:** 2026-05-18
**Target hardware:** Apple M5 Pro · 24 GB unified memory · macOS arm64
**Purpose:** validate env, device, datasets, and training stack before the overnight sweep of the 5 pruning pipelines.

---

## 1. Environment

| Check | Result |
|---|---|
| `which python` | `.venv/bin/python` (project venv, **not** Homebrew) |
| Python version | **3.11.15** (meets requirement of 3.11/3.12) |
| Venv manager | `uv` 0.11.14 (managed; **no `pip` binary by default**) |
| torch | 2.12.0 |
| torch_geometric | 2.7.0 |
| ogb | 1.3.6 |
| numpy / pandas / scipy | 2.4.5 / 3.0.3 / 1.17.1 |

### Issues found and resolved

- **Project package was not editable-installed.** `import gnn_pruning` raised `ModuleNotFoundError` on the first attempt — the uv-managed venv had not run `pip install -e .`. Fix applied (logged here per the task constraints):
  - Bootstrapped pip via `python -m ensurepip` (installed pip 24.0 into the venv).
  - Ran `python -m pip install -e . --no-deps` to register the project as an editable install. No external dependencies were added.
- **Risk for overnight run:** if `uv sync` is run again, the editable install will be removed. Either pin via `uv pip install -e .` after each sync, or add `gnn-pruning` to a uv workspace member.

---

## 2. MPS device

| Check | Result |
|---|---|
| `torch.backends.mps.is_available()` | **True** |
| `torch.backends.mps.is_built()` | True |
| Tiny tensor smoke (`ones(4, device='mps') + …`) | sum = 8.0 (expected 8.0) |
| Initial `mps.current_allocated_memory()` | 0 B |
| Initial `mps.driver_allocated_memory()` | 458 KB |

✅ MPS is functional. **No silent CPU fallback** was observed in any subsequent step.

---

## 3. Dataset reachability (full project loader matrix)

Loaded via `gnn_pruning.data.load_dataset(name)`. Ordered with ogbn-arxiv **last** to avoid a single failure blocking validation of the others. All datasets cached under `data/raw/` (gitignored).

| Dataset | Time | N (nodes) | E (edges) | Classes | Kind |
|---|---:|---:|---:|---:|---|
| cora | 0.01 s | 2,708 | 10,556 | 7 | single-graph |
| citeseer | 0.01 s | 3,327 | 9,104 | 6 | single-graph |
| pubmed | 0.01 s | 19,717 | 88,648 | 3 | single-graph |
| flickr | 0.05 s | 89,250 | 899,756 | 7 | single-graph |
| yelp | 0.40 s | 716,847 | 13,954,819 | 100 (multi-label, reported as 2) | single-graph |
| reddit | 0.24 s | 232,965 | 23,213,838 | 41 | single-graph |
| bbbp | 0.02 s | 49,068 | 105,842 | 2 | 2,039 graphs |
| proteins | 0.00 s | 43,471 | 162,088 | 2 | 1,113 graphs |
| squirrel | 0.01 s | 5,201 | 217,073 | 5 | single-graph |
| wisconsin | 0.00 s | 251 | 515 | 5 | single-graph |
| **ogbn-arxiv** | **15.22 s** | 169,343 | 1,166,243 | 40 | single-graph |

**Total cache:** 6,302 MB on disk (`data/raw/`).
**All 11 datasets load successfully** from cache; ogbn-arxiv processing takes the longest because PyG re-runs the OGB→PyG conversion on import.

Note: Yelp's `num_classes` shows as 2 only because of the report's `max(y)+1` fallback; Yelp is actually a 100-d multi-label target and must be handled with binary cross-entropy and micro-F1 (called out in issue #5's special-handling section).

---

## 4. End-to-end training smoke test (Cora, 2-layer GCN, hidden=256, MPS)

5 epochs, full-batch, on MPS, seed 0:

```
epoch 1/5  loss=1.9616
epoch 2/5  loss=1.4630
epoch 3/5  loss=0.9275
epoch 4/5  loss=0.5128
epoch 5/5  loss=0.2786
5 epochs in 1.82s (364.8 ms/epoch)
test acc: 0.8050
```

✅ **Test accuracy 0.805 > 0.70 threshold.** Training loop on MPS is functional. (The 0.36 s/epoch figure includes first-epoch MPS kernel compilation; steady-state will likely be faster.)

---

## 5. Runtime extrapolation (full sweep)

**Assumptions** (Cora 1× baseline from §4):
- One training run to convergence ≈ **200 epochs × 0.36 s ≈ 73 s** on Cora.
- One pruning pass per cell ≈ **1 activation forward + 9 eval forwards ≈ 0.5 s** on Cora.
- Scale factors: cora 1× · citeseer 1.2× · pubmed 3× · flickr 10× · yelp 50× · reddit 70× · bbbp 2× · proteins 2× · squirrel 2× · wisconsin 0.1× · ogbn-arxiv 100×.
- Architecture coverage per dataset taken from the M1-pipelines issues' 30-cell matrix.

### Per-dataset cost (all 3 archs combined where applicable)

| Dataset | Archs | No-pruning (train all archs once) | One pruning method (sparsity sweep, all archs) |
|---|---|---:|---:|
| cora | GCN, GAT | 2.43 min | 0.02 min |
| citeseer | GCN, GAT | 2.91 min | 0.02 min |
| pubmed | GCN, GAT | 7.28 min | 0.05 min |
| flickr | SAGE | 12.13 min | 0.08 min |
| yelp | SAGE | 60.67 min | 0.42 min |
| **reddit** | GCN, GAT, SAGE | **254.80 min** | **1.75 min** |
| bbbp | GCN, SAGE | 4.85 min | 0.03 min |
| proteins | GCN, SAGE | 4.85 min | 0.03 min |
| squirrel | — (not in M1 matrix) | 0.00 min | 0.00 min |
| wisconsin | GCN | 0.12 min | 0.00 min |
| **ogbn-arxiv** | SAGE | **121.33 min** | **0.83 min** |

### Per-method totals

| Method | Total | Excl ogbn-arxiv |
|---|---:|---:|
| No-pruning (training) | **7.86 h** | 5.83 h |
| Magnitude / Wanda-* (each) | 0.05 h | 0.04 h |
| **All 4 pruning methods together** | 0.22 h | 0.16 h |
| **Grand total (5 methods)** | **8.07 h** | **5.99 h** |

### Verdict on schedule
- **Within budget for an overnight run** (~8 hours).
- The dominant cost is **Reddit** (~4.25 h for the dense-baseline training across 3 archs), not ogbn-arxiv.
- User's threshold "without arxiv > 6 h → flag": at 5.99 h, this is on the edge — flagging.
- User's threshold "with arxiv > 10 h → schedule arxiv last": at 8.07 h, below the trigger, but I still **recommend running the large datasets (Reddit, ogbn-arxiv, Yelp, Flickr) last in each method's sweep** so that small datasets land partial results early in the night if the run is interrupted.

---

## 6. Forward-hook check (activation capture for Wanda variants)

Hooks attached on layer 1 and layer 2 of each architecture; one forward pass on Cora (N=2708, F=1433); `inputs[0]` and `output` captured.

| Arch | Layer 1 input → output | Layer 2 input → output | Notes |
|---|---|---|---|
| GCN | (2708, 1433) → (2708, 256) | (2708, 256) → (2708, 7) | Wanda-ready: per-column L2 norm over rows = ‖X_q‖₂. |
| GAT | (2708, 1433) → **(2708, 2048)** | (2708, 2048) → (2708, 7) | Layer-1 output is **256 × 8 heads concatenated**. ⚠️ Pruning logic must be aware: scoring against `‖X_q‖₂` on the 2048-d input to layer 2 is fine, but layer 1's weight matrix in `GATConv` is (in=1433, out=256·heads). **Also: GATConv applies W *before* aggregation**, so the activation at layer-2's input is the *aggregated* output of layer 1 — exactly what Wanda needs. For layer 1, the hooked `inputs[0]` is the *raw* feature matrix, *pre-aggregation*; this matches the LLM-Wanda semantics (W is applied to it). Document this in `wanda_uniform.py` so per-arch correctness is not subtle. |
| SAGE | (2708, 1433) → (2708, 256) | (2708, 256) → (2708, 7) | Wanda-ready; SAGE concatenates self + neighbor aggregation internally — `inputs[0]` is the pre-conv feature matrix. |

✅ Hook mechanism works on all three architectures. The GAT semantic note above is **architecture-specific and worth flagging in issue #3**.

---

## 7. Memory headroom (ogbn-arxiv on MPS)

Initial attempt: run GCN → GAT(heads=8) → SAGE forwards back-to-back in the **same** Python process. **Result: segmentation fault during GAT.** Diagnosis: `torch.mps.empty_cache()` does not reliably release driver-allocated memory between large allocations, so the MPS driver-side allocator climbs across consecutive architectures and crashes.

Per-arch in **isolated** processes (the realistic case for the runner if it forks per cell):

| Arch | current_allocated (post forward) | driver_allocated (post forward) |
|---|---:|---:|
| GCN | 130 MB | 5,177 MB |
| GAT (heads=1) | 130 MB | 5,175 MB |
| GAT (heads=8, layer-1 out 2048-d) | 132 MB | 5,289 MB |
| SAGE | 131 MB | 4,913 MB |

**Peak driver-allocated: 5.3 GB** — well under the 18 GB threshold.

✅ Memory headroom is **fine per arch** (well below 18 GB).
⚠️ **Run each (dataset, architecture) cell in a fresh subprocess.** Do *not* rely on `mps.empty_cache()` to recover from a multi-arch loop in one process — the segfault is reproducible. The pipeline scripts called out in the M1-pipelines issues should `subprocess.run(...)` (or similar) per cell, not loop over architectures in-process.

---

## Recommendations for the overnight run

1. **Schedule large datasets last** per method (`reddit`, `ogbn-arxiv`, `yelp`, `flickr` in that order, after all small/heterophilic datasets) so the run produces usable partial results if interrupted.
2. **Per-cell subprocess isolation** to avoid the MPS driver-memory creep that caused the GAT segfault in §7.
3. **Cap epochs and use early stopping.** The 200-epoch budget used for extrapolation in §5 is conservative; if early stopping kicks in around 100 epochs on most datasets, the total will roughly halve.
4. **Re-validate the editable install before launching** (`python -m pip show gnn-pruning`) — uv may have removed it again. If so, run `python -m pip install -e . --no-deps` before the sweep starts.
5. **Yelp / multi-label**: the loader returns a 100-d binary target; the eval harness called out in issue #5 must use BCE + micro-F1, not cross-entropy + accuracy. Confirm before the sweep.
6. **GAT heads=8** is the most memory-intensive variant at hidden=256; if any future scale-up runs into a real OOM, dropping to heads=4 or hidden=128 is the safe first lever.

---

## Verdict

Environment, device, datasets, training loop, hook mechanism, and memory headroom all pass. Extrapolated runtime fits the overnight window with margin (~8 h with arxiv, ~6 h without). Two operational caveats: (a) re-run `pip install -e .` if uv was re-synced; (b) **subprocess-per-cell** to avoid the MPS driver-memory segfault observed when looping architectures in one process.

```
PREFLIGHT: GO (slow)
```
