"""
utils/index_utils.py

Stage 2 utility module for TOP-FedTAMU+.

This file centralizes unlearning-index logic and, by default, follows the
original repository's return_unlearn_datasets() selection policy:

    helpful:
        1. sort influence descending
        2. take top int(0.31 * N) candidates
        3. randomly sample int(unlearn_prop * N) points from candidates

    harmful:
        1. sort influence ascending
        2. take top int(0.31 * N) candidates
        3. randomly sample int(unlearn_prop * N) points from candidates

    random:
        randomly sample int(unlearn_prop * N) points from the whole dataset

Why candidate_ratio=0.31?
-------------------------
The original code comments: "assume the maximum unlearning ratio is 0.3" and
uses candidate_no = int(0.31 * len(dataset_to_be_unlearn)). The extra 0.01 is
a small buffer so that unlearn_prop up to 0.3 can be sampled from the candidate
pool.

We also keep a "topk" policy as an optional Stage-1 reproducibility mode, but
the default is now "repo_candidate_random".

Supported modes
---------------
    random
    helpful
    harmful
    event / event_system
    event_mask

Important distinction
---------------------
index_criteria  -> how D_unlearn is selected
repair_criteria -> which objective is repaired

This module only handles index_criteria.
"""

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover
    OmegaConf = None

from utils import NewDataset


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def _select_cfg(cfg: Any, key: str, default: Any = None) -> Any:
    """Safe OmegaConf / dict / attribute selector."""
    if OmegaConf is not None:
        try:
            return OmegaConf.select(cfg, key, default=default)
        except Exception:
            pass

    cur = cfg
    for part in key.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return cur


def _as_numpy_index(index_like: Any) -> np.ndarray:
    """Convert an index-like object to a 1D int numpy array."""
    return np.asarray(index_like).astype(int).reshape(-1)


def _filter_valid_index(index: np.ndarray, n: int) -> np.ndarray:
    """Keep only valid row indices. Do not sort unless needed by caller."""
    index = _as_numpy_index(index)
    return index[(index >= 0) & (index < n)]


def _num_unlearn_samples(n: int, unlearn_prop: float) -> int:
    """
    Convert unlearn_prop to number of row-level samples.

    Original repository uses:
        unlearn_no = int(unlearn_prop * len(dataset))
    """
    if n <= 0:
        return 0
    return int(float(unlearn_prop) * n)


def build_remain_index(n: int, unlearn_index: np.ndarray) -> np.ndarray:
    """
    Build remain_index from row-level unlearn_index.

    Original repository uses a list comprehension preserving natural order:
        remain_index = [i for i in range(N) if i not in unlearn_index]
    """
    unlearn_index = _filter_valid_index(unlearn_index, n)
    mask = np.ones(n, dtype=bool)
    mask[unlearn_index] = False
    return np.arange(n, dtype=int)[mask]


# ---------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------
def make_subset_dataset(dataset: Any, index: np.ndarray) -> NewDataset:
    """Build a NewDataset subset while preserving scaling metadata."""
    index = _filter_valid_index(index, len(dataset))

    target_mean = getattr(dataset, "target_mean", 0)
    target_std = getattr(dataset, "target_std", 1)

    subset = NewDataset(
        dataset.feature[index],
        dataset.target[index],
        target_mean,
        target_std,
    )
    subset.is_scale = getattr(dataset, "is_scale", False)
    return subset


def split_dataset_by_index(
    dataset: Any,
    unlearn_index: np.ndarray,
) -> Tuple[NewDataset, NewDataset]:
    """
    Split a dataset into D_unlearn and D_remain by row-level indices.

    Returns:
        dataset_unlearn, dataset_remain
    """
    n = len(dataset)
    unlearn_index = _filter_valid_index(unlearn_index, n)
    remain_index = build_remain_index(n, unlearn_index)

    dataset_unlearn = make_subset_dataset(dataset, unlearn_index)
    dataset_remain = make_subset_dataset(dataset, remain_index)

    return dataset_unlearn, dataset_remain


def split_dataset_by_unlearn_object(
    dataset: Any,
    unlearn_object: Dict[str, Any],
) -> Tuple[NewDataset, NewDataset]:
    """Split using the object returned by load_unlearn_object()."""
    return split_dataset_by_index(dataset, unlearn_object["unlearn_index"])


# ---------------------------------------------------------------------
# Influence path and score loading
# ---------------------------------------------------------------------
def resolve_influence_path(
    cfg: Any,
    model_type: str,
    index_criteria: str,
) -> str:
    """
    Resolve the influence-score path.

    Original scripts pass cfg.model.influence_dir as a file in eval_unlearn.py
    and save criterion-specific influence arrays as:
        <cfg.influence_dir>/<model_type>_<criteria>.npy
    in gen_index.py.

    We check both conventions.
    """
    index_criteria = str(index_criteria)

    candidates = []

    influence_dir = _select_cfg(cfg, "influence_dir", None)
    if influence_dir is not None:
        candidates.append(os.path.join(str(influence_dir), f"{model_type}_{index_criteria}.npy"))

    model_influence = _select_cfg(cfg, "model.influence_dir", None)
    if model_influence is not None:
        model_influence = str(model_influence)
        if model_influence.endswith(".npy"):
            candidates.append(model_influence)
        else:
            candidates.append(os.path.join(model_influence, f"{model_type}_{index_criteria}.npy"))

    candidates.append(os.path.join("influence", f"{model_type}_{index_criteria}.npy"))

    checked = []
    for path in candidates:
        if path is None:
            continue
        path = os.path.normpath(path)
        checked.append(path)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Cannot find influence score file for "
        f"model_type={model_type}, index_criteria={index_criteria}.\n"
        "Checked paths:\n" + "\n".join(f"  - {p}" for p in checked)
    )


def load_influence_scores(
    cfg: Any,
    model_type: str,
    index_criteria: str,
) -> Tuple[np.ndarray, str]:
    """Load original TAMU influence scores."""
    path = resolve_influence_path(cfg, model_type, index_criteria)
    scores = np.load(path)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    return scores, path


# ---------------------------------------------------------------------
# Original-repo helpful / harmful / random selection
# ---------------------------------------------------------------------
def select_random_index_repo(
    n: int,
    unlearn_prop: float,
    random_seed: int,
) -> np.ndarray:
    """
    Original random policy:
        set_random_seed(config.data.random_seed)
        unlearn_no = int(unlearn_prop * N)
        np.random.choice(N, unlearn_no, replace=False)
    """
    unlearn_no = _num_unlearn_samples(n, unlearn_prop)
    np.random.seed(int(random_seed))
    return np.random.choice(n, unlearn_no, replace=False).astype(int)


def select_helpful_harmful_repo_candidate_random(
    scores: np.ndarray,
    n: int,
    unlearn_prop: float,
    mode: str,
    random_seed: int,
    candidate_ratio: float = 0.31,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Strictly follow original return_unlearn_datasets() policy for helpful/harmful:

        unlearn_no = int(unlearn_prop * N)
        candidate_no = int(0.31 * N)

        helpful:
            candidate_index = argsort(influences, descending=True)[:candidate_no]
        harmful:
            candidate_index = argsort(influences, descending=False)[:candidate_no]

        unlearn_index = np.random.choice(candidate_index, unlearn_no, replace=False)

    Notes:
        - candidate_no must be >= unlearn_no.
        - random_seed follows config.data.random_seed.
        - We use np.random.seed + np.random.choice to match repository behavior,
          not numpy.default_rng.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)

    if len(scores) != n:
        raise ValueError(
            f"Influence score length mismatch: len(scores)={len(scores)}, dataset length={n}."
        )

    mode = str(mode).lower()
    if mode not in ["helpful", "harmful"]:
        raise ValueError(f"mode must be helpful/harmful, got {mode}")

    unlearn_no = _num_unlearn_samples(n, unlearn_prop)
    candidate_no = int(float(candidate_ratio) * n)

    if candidate_no < unlearn_no:
        raise ValueError(
            f"candidate_no={candidate_no} is smaller than unlearn_no={unlearn_no}. "
            f"Increase candidate_ratio or decrease unlearn_prop."
        )

    if mode == "helpful":
        candidate_index = np.argsort(scores)[::-1][:candidate_no]
    else:
        candidate_index = np.argsort(scores)[:candidate_no]

    np.random.seed(int(random_seed))
    unlearn_index = np.random.choice(candidate_index, unlearn_no, replace=False).astype(int)

    # Original code prints the average/sum performance change on the candidate pool.
    # We record both candidate and selected stats for reproducibility.
    meta = {
        "selection_policy": "repo_candidate_random",
        "candidate_ratio": float(candidate_ratio),
        "candidate_no": int(candidate_no),
        "unlearn_no": int(unlearn_no),
        "candidate_score_sum": float(np.sum(scores[candidate_index])),
        "candidate_score_mean": float(np.mean(scores[candidate_index])),
        "selected_score_sum": float(np.sum(scores[unlearn_index])),
        "selected_score_mean": float(np.mean(scores[unlearn_index])),
        "random_seed": int(random_seed),
    }
    return unlearn_index, meta


def select_helpful_harmful_topk(
    scores: np.ndarray,
    n: int,
    unlearn_prop: float,
    mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Stage-1 reproducibility policy:
        helpful -> directly take top-k largest scores
        harmful -> directly take top-k smallest scores

    This is NOT the original repository policy. It is kept only to reproduce
    the Stage-1 fixedcrit tables.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)

    if len(scores) != n:
        raise ValueError(
            f"Influence score length mismatch: len(scores)={len(scores)}, dataset length={n}."
        )

    k = _num_unlearn_samples(n, unlearn_prop)
    mode = str(mode).lower()

    if mode == "helpful":
        index = np.argsort(scores)[-k:]
    elif mode == "harmful":
        index = np.argsort(scores)[:k]
    else:
        raise ValueError(f"mode must be helpful/harmful, got {mode}")

    meta = {
        "selection_policy": "topk_stage1_repro",
        "unlearn_no": int(k),
        "selected_score_sum": float(np.sum(scores[index])),
        "selected_score_mean": float(np.mean(scores[index])),
    }
    return index.astype(int), meta


# ---------------------------------------------------------------------
# Event index loading
# ---------------------------------------------------------------------
def top_fedtamu_dir(cfg: Any, model_type: str) -> str:
    """Return <cfg.simulation_dir>/<model_type>/top_fedtamu."""
    simulation_dir = str(_select_cfg(cfg, "simulation_dir", "simulation_result"))
    return os.path.join(simulation_dir, model_type, "top_fedtamu")


def resolve_event_paths(cfg: Any, model_type: str) -> Dict[str, str]:
    """Resolve files generated by gen_event_index.py."""
    base = top_fedtamu_dir(cfg, model_type)
    return {
        "base_dir": base,
        "unlearn_time_index": os.path.join(base, "unlearn_time_index.npy"),
        "unlearn_mask": os.path.join(base, "unlearn_mask.npy"),
        "unlearn_mask_system": os.path.join(base, "unlearn_mask_system.npy"),
        "score_node": os.path.join(base, "score_node.npy"),
        "score_time": os.path.join(base, "score_time.npy"),
        "score_event": os.path.join(base, "score_event.npy"),
    }


def load_event_system_index(
    cfg: Any,
    model_type: str,
    n: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load event-system row-level index.

    event_system means:
        if time t is selected, all bus outputs at row t are treated as the
        unlearned system snapshot.
    """
    paths = resolve_event_paths(cfg, model_type)
    time_path = paths["unlearn_time_index"]

    if not os.path.exists(time_path):
        raise FileNotFoundError(
            f"Event-system index not found: {time_path}\n"
            "Please run gen_event_index.py first."
        )

    unlearn_index = _filter_valid_index(np.load(time_path), n)

    metadata = {
        "event_base_dir": paths["base_dir"],
        "event_time_index_path": time_path,
        "index_granularity": "row_time_snapshot",
        "mask_available": os.path.exists(paths["unlearn_mask"]),
        "system_mask_available": os.path.exists(paths["unlearn_mask_system"]),
    }
    return unlearn_index, metadata


def load_event_mask_index(
    cfg: Any,
    model_type: str,
    n: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load event-mask index.

    Returns:
        row-level unlearn_index = all rows t where any bus is masked
        metadata["unlearn_mask"] = full bus-time mask M[t, k]
    """
    paths = resolve_event_paths(cfg, model_type)
    mask_path = paths["unlearn_mask"]

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Event mask not found: {mask_path}\n"
            "Please run gen_event_index.py first."
        )

    unlearn_mask = np.asarray(np.load(mask_path)).astype(bool)

    if unlearn_mask.ndim != 2:
        raise ValueError(f"unlearn_mask should be 2D [time, bus], got shape={unlearn_mask.shape}")

    row_index = np.where(unlearn_mask.any(axis=1))[0]
    row_index = _filter_valid_index(row_index, n)

    system_mask = None
    if os.path.exists(paths["unlearn_mask_system"]):
        system_mask = np.asarray(np.load(paths["unlearn_mask_system"])).astype(bool)

    metadata = {
        "event_base_dir": paths["base_dir"],
        "event_mask_path": mask_path,
        "system_mask_path": paths["unlearn_mask_system"] if system_mask is not None else None,
        "index_granularity": "bus_time_mask",
        "unlearn_mask": unlearn_mask,
        "unlearn_mask_system": system_mask,
        "num_masked_points": int(unlearn_mask.sum()),
        "num_masked_rows": int(len(row_index)),
    }
    return row_index, metadata


# ---------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------
def load_unlearn_object(
    cfg: Any,
    model_type: str,
    dataset: Any,
    index_mode: Optional[str] = None,
    index_criteria: Optional[str] = None,
    unlearn_prop: Optional[float] = None,
    random_seed: Optional[int] = None,
    selection_policy: Optional[str] = None,
    candidate_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Load or generate a unified unlearning-index object.

    Parameters
    ----------
    index_mode:
        random / helpful / harmful / event / event_system / event_mask.
    index_criteria:
        mse / mape / cost, used for helpful/harmful influence selection.
    selection_policy:
        helpful/harmful only.
        - "repo_candidate_random": original repository policy. DEFAULT.
        - "topk": Stage-1 reproducibility policy.
    candidate_ratio:
        helpful/harmful repo policy only. DEFAULT 0.31.

    Returns
    -------
    dict:
        unlearn_index
        remain_index
        index_mode
        index_criteria
        unlearn_prop
        metadata
    """
    n = len(dataset)

    if index_mode is None:
        index_mode = _select_cfg(
            cfg,
            "index_mode",
            _select_cfg(cfg, "unlearn_mode", "random"),
        )

    if index_criteria is None:
        index_criteria = _select_cfg(
            cfg,
            "index_criteria",
            _select_cfg(cfg, "criteria", "mse"),
        )

    if unlearn_prop is None:
        unlearn_prop = float(_select_cfg(cfg, "unlearn_prop", 0.2))

    if random_seed is None:
        random_seed = int(_select_cfg(cfg, "data.random_seed", 0))

    if selection_policy is None:
        selection_policy = _select_cfg(
            cfg,
            "selection_policy",
            _select_cfg(cfg, "index_selection_policy", "repo_candidate_random"),
        )

    if candidate_ratio is None:
        candidate_ratio = float(_select_cfg(cfg, "candidate_ratio", 0.31))

    mode = str(index_mode).lower()
    selection_policy = str(selection_policy).lower()

    metadata: Dict[str, Any] = {
        "requested_index_mode": mode,
        "index_criteria": str(index_criteria),
        "unlearn_prop": float(unlearn_prop),
        "dataset_length": int(n),
        "selection_policy": selection_policy,
        "candidate_ratio": float(candidate_ratio),
    }

    # ------------------------------------------------------------
    # random
    # ------------------------------------------------------------
    if mode == "random":
        unlearn_index = select_random_index_repo(
            n=n,
            unlearn_prop=unlearn_prop,
            random_seed=random_seed,
        )
        metadata.update(
            {
                "selection": "random",
                "random_seed": int(random_seed),
                "num_unlearn_rows": int(len(unlearn_index)),
            }
        )

    # ------------------------------------------------------------
    # helpful / harmful
    # ------------------------------------------------------------
    elif mode in ["helpful", "harmful"]:
        scores, score_path = load_influence_scores(cfg, model_type, index_criteria)

        if selection_policy in ["repo", "repo_candidate_random", "candidate_random"]:
            unlearn_index, sel_meta = select_helpful_harmful_repo_candidate_random(
                scores=scores,
                n=n,
                unlearn_prop=unlearn_prop,
                mode=mode,
                random_seed=random_seed,
                candidate_ratio=candidate_ratio,
            )
        elif selection_policy in ["topk", "stage1_topk", "topk_stage1_repro"]:
            unlearn_index, sel_meta = select_helpful_harmful_topk(
                scores=scores,
                n=n,
                unlearn_prop=unlearn_prop,
                mode=mode,
            )
        else:
            raise ValueError(
                f"Unsupported selection_policy={selection_policy}. "
                "Use repo_candidate_random or topk."
            )

        metadata.update(
            {
                "selection": mode,
                "score_path": score_path,
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
                "score_mean": float(np.mean(scores)),
                "score_std": float(np.std(scores)),
                "num_unlearn_rows": int(len(unlearn_index)),
            }
        )
        metadata.update(sel_meta)

    # ------------------------------------------------------------
    # event / event_system
    # ------------------------------------------------------------
    elif mode in ["event", "event_system"]:
        unlearn_index, event_meta = load_event_system_index(cfg, model_type, n)
        metadata.update(event_meta)
        metadata.update(
            {
                "selection": "event_system",
                "num_unlearn_rows": int(len(unlearn_index)),
            }
        )

    # ------------------------------------------------------------
    # event_mask
    # ------------------------------------------------------------
    elif mode == "event_mask":
        unlearn_index, event_meta = load_event_mask_index(cfg, model_type, n)
        metadata.update(event_meta)
        metadata.update(
            {
                "selection": "event_mask",
                "num_unlearn_rows": int(len(unlearn_index)),
                "warning": (
                    "event_mask returns row-level indices plus a bus-time mask. "
                    "Repo-style DatasetWithWeight only uses row-level indices; "
                    "true mask-aware repair must consume metadata['unlearn_mask']."
                ),
            }
        )

    else:
        raise ValueError(
            f"Unsupported index_mode={index_mode}. "
            "Supported modes: random, helpful, harmful, event, event_system, event_mask."
        )

    unlearn_index = _filter_valid_index(unlearn_index, n)
    remain_index = build_remain_index(n, unlearn_index)

    return {
        "unlearn_index": unlearn_index,
        "remain_index": remain_index,
        "index_mode": mode,
        "index_criteria": str(index_criteria),
        "unlearn_prop": float(unlearn_prop),
        "metadata": metadata,
    }


def load_and_split_unlearn_datasets(
    cfg: Any,
    model_type: str,
    dataset: Any,
    index_mode: Optional[str] = None,
    index_criteria: Optional[str] = None,
    unlearn_prop: Optional[float] = None,
    random_seed: Optional[int] = None,
    selection_policy: Optional[str] = None,
    candidate_ratio: Optional[float] = None,
) -> Tuple[NewDataset, NewDataset, Dict[str, Any]]:
    """
    Convenience wrapper:
        load_unlearn_object() + split_dataset_by_index()
    """
    obj = load_unlearn_object(
        cfg=cfg,
        model_type=model_type,
        dataset=dataset,
        index_mode=index_mode,
        index_criteria=index_criteria,
        unlearn_prop=unlearn_prop,
        random_seed=random_seed,
        selection_policy=selection_policy,
        candidate_ratio=candidate_ratio,
    )

    dataset_unlearn, dataset_remain = split_dataset_by_index(
        dataset,
        obj["unlearn_index"],
    )

    return dataset_unlearn, dataset_remain, obj


def save_unlearn_object(
    obj: Dict[str, Any],
    save_dir: str,
    prefix: str = "",
) -> None:
    """
    Save unified unlearning object arrays for reproducibility.

    Saved files:
        <prefix>unlearn_index.npy
        <prefix>remain_index.npy
        <prefix>unlearn_metadata.npy

    If event_mask metadata exists, also save:
        <prefix>unlearn_mask.npy
        <prefix>unlearn_mask_system.npy
    """
    os.makedirs(save_dir, exist_ok=True)

    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"

    np.save(os.path.join(save_dir, f"{prefix}unlearn_index.npy"), obj["unlearn_index"])
    np.save(os.path.join(save_dir, f"{prefix}remain_index.npy"), obj["remain_index"])
    np.save(
        os.path.join(save_dir, f"{prefix}unlearn_metadata.npy"),
        obj["metadata"],
        allow_pickle=True,
    )

    metadata = obj.get("metadata", {})
    if isinstance(metadata, dict):
        if metadata.get("unlearn_mask", None) is not None:
            np.save(os.path.join(save_dir, f"{prefix}unlearn_mask.npy"), metadata["unlearn_mask"])
        if metadata.get("unlearn_mask_system", None) is not None:
            np.save(os.path.join(save_dir, f"{prefix}unlearn_mask_system.npy"), metadata["unlearn_mask_system"])


def load_saved_unlearn_object(
    dataset: Any,
    load_dir: str,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Load a previously saved unlearning split and rebuild dataset subsets.

    Expected files:
        <prefix>unlearn_index.npy
        <prefix>remain_index.npy
        <prefix>unlearn_metadata.npy
    """
    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"

    unlearn_index = np.load(os.path.join(load_dir, f"{prefix}unlearn_index.npy"), allow_pickle=True)
    remain_index = np.load(os.path.join(load_dir, f"{prefix}remain_index.npy"), allow_pickle=True)
    metadata = np.load(
        os.path.join(load_dir, f"{prefix}unlearn_metadata.npy"),
        allow_pickle=True,
    ).item()

    n = len(dataset)
    unlearn_index = _filter_valid_index(np.asarray(unlearn_index, dtype=int), n)
    remain_index = _filter_valid_index(np.asarray(remain_index, dtype=int), n)

    dataset_unlearn = make_subset_dataset(dataset, unlearn_index)
    dataset_remain = make_subset_dataset(dataset, remain_index)

    return {
        "dataset_unlearn": dataset_unlearn,
        "dataset_remain": dataset_remain,
        "unlearn_object": {
            "unlearn_index": unlearn_index,
            "remain_index": remain_index,
            "index_mode": str(metadata.get("index_mode", "reused")),
            "index_criteria": str(metadata.get("index_criteria", "")),
            "unlearn_prop": float(metadata.get("unlearn_prop", len(unlearn_index) / max(n, 1))),
            "metadata": {
                **metadata,
                "reused_from_dir": os.path.abspath(load_dir),
            },
        },
    }
