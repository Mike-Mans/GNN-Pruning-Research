"""Dataset loaders for the GNN-Pruning-Research project.

One loader per benchmark named in `proposal.tex` / `README.md` (Ethereum
deferred). All datasets download on first use and cache under `data/raw/`
(gitignored). Callers should use `load_dataset(name)` rather than the
per-dataset functions directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Union

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import (
    Actor,
    Amazon,
    Coauthor,
    Flickr,
    MoleculeNet,
    Planetoid,
    Reddit2,
    TUDataset,
    WebKB,
    WikipediaNetwork,
    Yelp,
)

DatasetLike = Union[Data, InMemoryDataset]
TaskType = Literal["node-classification", "graph-classification"]
Homophily = Literal["homophilic", "heterophilic"]

DEFAULT_ROOT = Path("data/raw")


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    task: TaskType
    homophily: Homophily
    source: str
    citation: str
    multi_label: bool = False


def _subdir(root: Union[str, Path], name: str) -> str:
    return str(Path(root) / name)


def _allowlist_ogb_globals() -> None:
    import torch
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage

    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])


def load_cora(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Planetoid(root=_subdir(root, "Planetoid"), name="Cora")[0]


def load_citeseer(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Planetoid(root=_subdir(root, "Planetoid"), name="Citeseer")[0]


def load_pubmed(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Planetoid(root=_subdir(root, "Planetoid"), name="Pubmed")[0]


def load_flickr(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Flickr(root=_subdir(root, "Flickr"))[0]


def load_arxiv(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    from ogb.nodeproppred import PygNodePropPredDataset

    _allowlist_ogb_globals()
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=_subdir(root, "OGB"))
    data = dataset[0]
    data.split_idx = dataset.get_idx_split()
    data.num_classes = dataset.num_classes
    return data


def load_products(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    from ogb.nodeproppred import PygNodePropPredDataset

    _allowlist_ogb_globals()
    dataset = PygNodePropPredDataset(name="ogbn-products", root=_subdir(root, "OGB"))
    data = dataset[0]
    data.split_idx = dataset.get_idx_split()
    data.num_classes = dataset.num_classes
    return data


def load_ogbn_proteins(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    from ogb.nodeproppred import PygNodePropPredDataset

    _allowlist_ogb_globals()
    dataset = PygNodePropPredDataset(name="ogbn-proteins", root=_subdir(root, "OGB"))
    data = dataset[0]
    data.split_idx = dataset.get_idx_split()
    data.num_classes = dataset.num_classes
    # ogbn-proteins is multi-label (112 binary tasks); flag is consumed in eval.
    return data


def load_yelp(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Yelp(root=_subdir(root, "Yelp"))[0]


def load_yelpchi(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    """YelpChi fraud-detection graph.

    Available in PyG via `torch_geometric.datasets.YelpChi` in 2.5+. If the
    PyG version on this machine does not ship it, we surface a clear error
    rather than silently falling back.
    """
    try:
        from torch_geometric.datasets import YelpChi  # type: ignore
    except ImportError as e:  # pragma: no cover - environment-specific
        raise ImportError(
            "YelpChi loader requires torch_geometric.datasets.YelpChi "
            "(PyG >= 2.5). Pin a newer torch_geometric or remove the "
            "yelpchi cell from the active config."
        ) from e
    return YelpChi(root=_subdir(root, "YelpChi"))[0]


def load_reddit(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Reddit2(root=_subdir(root, "Reddit2"))[0]


def load_bbbp(root: Union[str, Path] = DEFAULT_ROOT) -> InMemoryDataset:
    return MoleculeNet(root=_subdir(root, "MoleculeNet"), name="BBBP")


def load_proteins(root: Union[str, Path] = DEFAULT_ROOT) -> InMemoryDataset:
    return TUDataset(root=_subdir(root, "TUDataset"), name="PROTEINS")


def load_squirrel(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return WikipediaNetwork(
        root=_subdir(root, "WikipediaNetwork"),
        name="squirrel",
        geom_gcn_preprocess=True,
    )[0]


def load_wisconsin(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return WebKB(root=_subdir(root, "WebKB"), name="Wisconsin")[0]


def load_cornell(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return WebKB(root=_subdir(root, "WebKB"), name="Cornell")[0]


def load_texas(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return WebKB(root=_subdir(root, "WebKB"), name="Texas")[0]


def load_actor(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Actor(root=_subdir(root, "Actor"))[0]


def load_cs(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Coauthor(root=_subdir(root, "Coauthor"), name="CS")[0]


def load_physics(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Coauthor(root=_subdir(root, "Coauthor"), name="Physics")[0]


def load_photo(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Amazon(root=_subdir(root, "Amazon"), name="Photo")[0]


def load_computers(root: Union[str, Path] = DEFAULT_ROOT) -> Data:
    return Amazon(root=_subdir(root, "Amazon"), name="Computers")[0]


DATASET_REGISTRY: dict[str, Callable[..., DatasetLike]] = {
    "cora": load_cora,
    "citeseer": load_citeseer,
    "pubmed": load_pubmed,
    "flickr": load_flickr,
    "ogbn-arxiv": load_arxiv,
    "ogbn-products": load_products,
    "ogbn-proteins": load_ogbn_proteins,
    "yelp": load_yelp,
    "yelpchi": load_yelpchi,
    "reddit": load_reddit,
    "bbbp": load_bbbp,
    "proteins": load_proteins,
    "squirrel": load_squirrel,
    "wisconsin": load_wisconsin,
    "cornell": load_cornell,
    "texas": load_texas,
    "actor": load_actor,
    "cs": load_cs,
    "physics": load_physics,
    "photo": load_photo,
    "computers": load_computers,
}


DATASET_META: dict[str, DatasetMeta] = {
    "cora": DatasetMeta("Cora", "node-classification", "homophilic",
                        "Planetoid", "Sen et al., AI Magazine 2008"),
    "citeseer": DatasetMeta("Citeseer", "node-classification", "homophilic",
                            "Planetoid", "Sen et al., AI Magazine 2008"),
    "pubmed": DatasetMeta("Pubmed", "node-classification", "homophilic",
                          "Planetoid", "Sen et al., AI Magazine 2008"),
    "flickr": DatasetMeta("Flickr", "node-classification", "homophilic",
                          "GraphSAINT", "Zeng et al., ICLR 2020"),
    "ogbn-arxiv": DatasetMeta("ogbn-arxiv", "node-classification", "homophilic",
                              "OGB", "Hu et al., NeurIPS 2020"),
    "ogbn-products": DatasetMeta("ogbn-products", "node-classification", "homophilic",
                                 "OGB", "Hu et al., NeurIPS 2020"),
    "ogbn-proteins": DatasetMeta("ogbn-proteins", "node-classification", "homophilic",
                                 "OGB", "Hu et al., NeurIPS 2020", multi_label=True),
    "yelp": DatasetMeta("Yelp", "node-classification", "homophilic",
                        "GraphSAINT", "Zeng et al., ICLR 2020", multi_label=True),
    "yelpchi": DatasetMeta("YelpChi", "node-classification", "heterophilic",
                           "DGFraud / Dou et al., CIKM 2020",
                           "Dou et al., CIKM 2020"),
    "reddit": DatasetMeta("Reddit", "node-classification", "homophilic",
                          "Reddit2 (GraphSAINT variant)",
                          "Hamilton et al., NeurIPS 2017"),
    "bbbp": DatasetMeta("BBBP", "graph-classification", "homophilic",
                        "MoleculeNet", "Wu et al., Chem. Sci. 2018"),
    "proteins": DatasetMeta("PROTEINS", "graph-classification", "homophilic",
                            "TUDataset", "Borgwardt et al., Bioinformatics 2005"),
    "squirrel": DatasetMeta("Squirrel", "node-classification", "heterophilic",
                            "WikipediaNetwork (geom-gcn split)",
                            "Rozemberczki et al., 2021"),
    "wisconsin": DatasetMeta("Wisconsin", "node-classification", "heterophilic",
                             "WebKB", "Pei et al., ICLR 2020"),
    "cornell": DatasetMeta("Cornell", "node-classification", "heterophilic",
                           "WebKB", "Pei et al., ICLR 2020"),
    "texas": DatasetMeta("Texas", "node-classification", "heterophilic",
                         "WebKB", "Pei et al., ICLR 2020"),
    "actor": DatasetMeta("Actor", "node-classification", "heterophilic",
                         "PyG Actor", "Pei et al., ICLR 2020"),
    "cs": DatasetMeta("CS", "node-classification", "homophilic",
                      "Coauthor", "Shchur et al., 2018"),
    "physics": DatasetMeta("Physics", "node-classification", "homophilic",
                           "Coauthor", "Shchur et al., 2018"),
    "photo": DatasetMeta("Photo", "node-classification", "homophilic",
                         "Amazon", "Shchur et al., 2018"),
    "computers": DatasetMeta("Computers", "node-classification", "homophilic",
                             "Amazon", "Shchur et al., 2018"),
}


def load_dataset(
    name: str, root: Union[str, Path] = DEFAULT_ROOT
) -> DatasetLike:
    key = name.lower().strip()
    if key not in DATASET_REGISTRY:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset {name!r}. Valid names: {valid}")
    return DATASET_REGISTRY[key](root)
