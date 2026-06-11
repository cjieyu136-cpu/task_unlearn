"""
utils/fed_cache_utils.py

Small cache helpers for the topology-local-fusion NN path.

This cache is intentionally simple:
    - cache only deterministic frozen-feature artifacts
    - avoid caching any repair / unlearning outputs
    - store metadata alongside arrays for traceability
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from utils import NewDataset


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def stable_cache_key(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_topology_fusion_cache_dir(
    simulation_dir: str,
    model_type: str,
    payload: Dict[str, Any],
) -> Path:
    key = stable_cache_key(payload)
    return Path(str(simulation_dir)) / "_fed_cache" / str(model_type) / "topology_local_fusion" / key


def load_topology_fusion_cache(
    cache_dir: Path,
    target_mean,
    target_std,
    train_target,
    test_target,
    is_scale: bool,
):
    meta_path = cache_dir / "meta.json"
    train_feature_path = cache_dir / "train_feature.npy"
    test_feature_path = cache_dir / "test_feature.npy"
    bus_groups_path = cache_dir / "bus_groups.npy"
    client_graph_path = cache_dir / "client_graph.npy"
    encoder_upload_path = cache_dir / "encoder_upload_summary.csv"
    client_embedding_path = cache_dir / "client_embedding_summary.csv"
    fusion_feature_path = cache_dir / "fusion_feature_summary.csv"

    required = [
        meta_path,
        train_feature_path,
        test_feature_path,
        bus_groups_path,
        client_graph_path,
        encoder_upload_path,
        client_embedding_path,
        fusion_feature_path,
    ]
    if not all(p.exists() for p in required):
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_feature = np.load(train_feature_path)
    test_feature = np.load(test_feature_path)
    bus_groups_raw = np.load(bus_groups_path, allow_pickle=True)
    client_graph = np.load(client_graph_path)

    dataset_train_affine = NewDataset(torch.tensor(train_feature).float(), train_target, target_mean, target_std)
    dataset_test_affine = NewDataset(torch.tensor(test_feature).float(), test_target, target_mean, target_std)
    dataset_train_affine.is_scale = bool(is_scale)
    dataset_test_affine.is_scale = bool(is_scale)

    return {
        "meta": meta,
        "dataset_train_affine": dataset_train_affine,
        "dataset_test_affine": dataset_test_affine,
        "bus_groups": [np.asarray(x, dtype=int) for x in bus_groups_raw.tolist()],
        "bus_axis": int(meta["bus_axis"]),
        "client_graph": client_graph,
        "encoder_upload_summary": pd.read_csv(encoder_upload_path),
        "client_embedding_summary": pd.read_csv(client_embedding_path),
        "fusion_feature_summary": pd.read_csv(fusion_feature_path),
    }


def save_topology_fusion_cache(
    cache_dir: Path,
    *,
    meta: Dict[str, Any],
    dataset_train_affine,
    dataset_test_affine,
    bus_groups,
    client_graph,
    encoder_upload_summary: pd.DataFrame,
    client_embedding_summary: pd.DataFrame,
    fusion_feature_summary: pd.DataFrame,
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    (cache_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    np.save(cache_dir / "train_feature.npy", _to_numpy(dataset_train_affine.feature))
    np.save(cache_dir / "test_feature.npy", _to_numpy(dataset_test_affine.feature))
    np.save(cache_dir / "bus_groups.npy", np.asarray([np.asarray(g, dtype=int) for g in bus_groups], dtype=object), allow_pickle=True)
    np.save(cache_dir / "client_graph.npy", np.asarray(client_graph, dtype=float))
    encoder_upload_summary.to_csv(cache_dir / "encoder_upload_summary.csv", index=False)
    client_embedding_summary.to_csv(cache_dir / "client_embedding_summary.csv", index=False)
    fusion_feature_summary.to_csv(cache_dir / "fusion_feature_summary.csv", index=False)
