"""
utils/fed_server_runtime.py

Stage 3H FedServer runtime abstraction.

Supported feature modes:
    precomputed_local_cache
    local_frozen_backbone
    topology_local_fusion
    local_mask_fusion / fusion_topology_only

This is a single-machine client/server simulation boundary, not secure
aggregation or distributed networking.
"""

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from utils.fed_vjp_utils import cloud_criterion_gradient_scaled_for_dataset
from utils.fed_local_head import extract_affine_weight_bias, canonical_vec_to_model_order
from utils.fed_client_runtime import (
    build_clients_from_affine_dataset,
    build_clients_from_raw_dataset,
)
from utils.fed_topology_nn import build_topology_local_fusion_datasets


class FedServer:
    def __init__(self, cfg, model, dataset_reference, clients: List[Any], bus_groups, feature_mode: str):
        self.cfg = cfg
        self.model = model
        self.dataset_reference = dataset_reference
        self.clients = list(clients)
        self.bus_groups = bus_groups
        self.feature_mode = str(feature_mode)
        self.messages = []

        self.feature_dim = int(dataset_reference.feature.shape[1])
        self.output_dim = int(dataset_reference.target.shape[1])

    def _record(self, message):
        self.messages.append(message)

    def prepare_client_features(self) -> pd.DataFrame:
        rows = []
        for client in self.clients:
            result = client.compute_local_features()
            self._record(result["message"])
            rows.append({
                **client.local_metadata(include_private_shapes=True),
                "feature_shape": str(result["feature_shape"]),
                "feature_norm": result["feature_norm"],
            })
        return pd.DataFrame(rows)

    def collect_predictions(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        n_rows = len(self.dataset_reference) if indices is None else len(indices)
        full = np.zeros((n_rows, self.output_dim), dtype=np.float64)

        for client in self.clients:
            result = client.predict_slice(indices=indices)
            self._record(result["message"])
            full[:, client.bus_indices] = result["prediction"]

        return full

    def compute_cloud_gy_and_dispatch(
        self,
        repair_criteria: str = "cost",
        split_name: str = "test",
        batch_size: int = 128,
        device: Optional[str] = None,
    ):
        cloud = cloud_criterion_gradient_scaled_for_dataset(
            cfg=self.cfg,
            model=self.model,
            dataset=self.dataset_reference,
            repair_criteria=repair_criteria,
            batch_size=batch_size,
            device=device,
        )
        gy = np.asarray(cloud["gy_scaled"], dtype=np.float64)

        for client in self.clients:
            msg = client.receive_gy_slice(gy[:, client.bus_indices], split_name=split_name)
            self._record(msg)

        return {
            "gy_scaled": gy,
            "cloud_info": cloud["info"],
        }

    def compute_runtime_block_influence(
        self,
        remain_indices: Optional[np.ndarray] = None,
        score_indices: Optional[np.ndarray] = None,
        split_name: str = "test",
        block_damping: float = 1e-8,
    ):
        W_grad_full = np.zeros((self.output_dim, self.feature_dim), dtype=np.float64)
        b_grad_full = np.zeros((self.output_dim,), dtype=np.float64)

        W_M_full = np.zeros((self.output_dim, self.feature_dim), dtype=np.float64)
        b_M_full = np.zeros((self.output_dim,), dtype=np.float64)

        score_total = None

        grad_rows = []
        M_rows = []
        score_rows = []

        for client in self.clients:
            grad = client.compute_criterion_gradient(indices=None, split_name=split_name)
            self._record(grad["message"])

            M = client.compute_block_ihvp(
                W_grad=grad["W_grad"],
                b_grad=grad["b_grad"],
                indices=remain_indices,
                damping=block_damping,
            )
            self._record(M["message"])

            score = client.compute_score_contribution(
                W_M=M["W_M"],
                b_M=M["b_M"],
                indices=score_indices,
                normalize_by_n=True,
            )
            self._record(score["message"])

            buses = client.bus_indices
            W_grad_full[buses, :] = grad["W_grad"]
            b_grad_full[buses] = grad["b_grad"]

            W_M_full[buses, :] = M["W_M"]
            b_M_full[buses] = M["b_M"]

            if score_total is None:
                score_total = np.zeros_like(score["score"], dtype=np.float64)
            score_total = score_total + score["score"]

            meta = client.local_metadata()
            grad_rows.append({
                **meta,
                "g_k_norm": float(np.linalg.norm(grad["grad_vec"])),
                "g_k_elements": int(grad["grad_vec"].size),
            })
            M_rows.append({
                **meta,
                "M_k_norm": float(np.linalg.norm(M["M_vec"])),
                "M_k_elements": int(M["M_vec"].size),
                "H_condition": float(M["H_condition"]),
                "solver": M["solver"],
            })
            score_rows.append({
                **meta,
                "score_min": float(np.min(score["score"])),
                "score_max": float(np.max(score["score"])),
                "score_mean": float(np.mean(score["score"])),
                "score_std": float(np.std(score["score"])),
            })

        grad_canonical = np.concatenate([W_grad_full.reshape(-1), b_grad_full.reshape(-1)])
        M_canonical = np.concatenate([W_M_full.reshape(-1), b_M_full.reshape(-1)])

        grad_model_order = canonical_vec_to_model_order(
            model=self.model,
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            canonical_vec=grad_canonical,
        )
        M_model_order = canonical_vec_to_model_order(
            model=self.model,
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            canonical_vec=M_canonical,
        )

        return {
            "grad_canonical": grad_canonical,
            "M_canonical": M_canonical,
            "grad_model_order": grad_model_order,
            "M_model_order": M_model_order,
            "scores": score_total,
            "grad_client_summary": pd.DataFrame(grad_rows),
            "M_client_summary": pd.DataFrame(M_rows),
            "score_client_summary": pd.DataFrame(score_rows),
            "payload_summary": self.payload_summary(),
        }

    def payload_summary(self) -> pd.DataFrame:
        rows = []
        for msg in self.messages:
            rows.append({
                "client_id": msg.client_id,
                "message_type": msg.message_type,
                "payload_shape": msg.payload_shape,
                "num_elements": msg.num_elements,
                **{f"meta_{k}": v for k, v in msg.metadata.items()},
            })
        return pd.DataFrame(rows)


def _get_affine_head(model, dataset_reference):
    feature_dim = int(dataset_reference.feature.shape[1])
    output_dim = int(dataset_reference.target.shape[1])
    return extract_affine_weight_bias(
        model,
        feature_dim=feature_dim,
        output_dim=output_dim,
    )


def build_runtime_server_from_affine_context(
    cfg,
    ctx,
    dataset_for_clients,
    feature_mode: str = "precomputed_local_cache",
):
    model = ctx["model_ori"]
    affine = _get_affine_head(model, dataset_for_clients)

    clients = build_clients_from_affine_dataset(
        dataset=dataset_for_clients,
        W_out_in=affine["W_out_in"],
        b=affine["b"],
        bus_groups=ctx["bus_groups"],
        feature_mode=feature_mode,
    )

    return FedServer(
        cfg=cfg,
        model=model,
        dataset_reference=dataset_for_clients,
        clients=clients,
        bus_groups=ctx["bus_groups"],
        feature_mode=feature_mode,
    )


def build_runtime_server_from_raw_context(
    cfg,
    ctx,
    raw_dataset_for_clients,
    target_dataset_reference,
    frozen_backbone,
    feature_batch_size: int = 128,
    device: str = "cpu",
):
    """
    H2-2 server/client construction:
        clients compute local features from raw input + frozen backbone.

    target_dataset_reference is used as the repo-compatible reference for target
    and server-side OPF utilities; clients do not receive its feature matrix.
    """
    model = ctx["model_ori"]
    affine = _get_affine_head(model, target_dataset_reference)

    clients = build_clients_from_raw_dataset(
        raw_dataset=raw_dataset_for_clients,
        target_dataset=target_dataset_reference,
        frozen_backbone=frozen_backbone,
        W_out_in=affine["W_out_in"],
        b=affine["b"],
        bus_groups=ctx["bus_groups"],
        feature_batch_size=feature_batch_size,
        device=device,
    )

    return FedServer(
        cfg=cfg,
        model=model,
        dataset_reference=target_dataset_reference,
        clients=clients,
        bus_groups=ctx["bus_groups"],
        feature_mode="local_frozen_backbone",
    )


def build_runtime_topology_servers_from_raw_context(
    cfg,
    ctx,
    raw_train_dataset,
    raw_test_dataset,
    frozen_backbone,
    device: str = "cpu",
):
    """
    Runtime bridge for the paper-aligned path:

        client local raw input
        -> client local embedding
        -> server-side topology fusion / aggregation layout
        -> affine-head runtime clients

    Returns train/test servers plus the runtime-generated affine datasets and
    summary tables from the embedding/fusion stage.
    """
    extra = ctx.get("extra_info", {})
    common = ctx["common"]

    partition_enabled = bool(extra.get("topology_partition_enabled", False))
    partition_mode = str(extra.get("topology_partition_mode", "topology"))
    tau = float(extra.get("topology_tau", 1.0))
    encoder_self_weight = float(extra.get("encoder_self_weight", 3.0))
    encoder_mode = str(extra.get("encoder_topology_mode", "topology"))
    fusion_alpha = float(extra.get("fusion_topology_alpha", 0.5))
    feature_layout = str(extra.get("server_feature_layout", extra.get("fusion_feature_layout", "secure_agg_mean")))
    use_smoothing_residual = bool(extra.get("fusion_use_smoothing_residual", True))
    normalize_feature = bool(extra.get("fusion_normalize_feature", True))

    fusion_ctx = build_topology_local_fusion_datasets(
        dataset_sensitive=raw_train_dataset,
        dataset_test=raw_test_dataset,
        core_model=frozen_backbone,
        A_grid=ctx["A_grid"],
        num_bus_clients=common["num_bus_clients"],
        bus_groups_override=common["bus_groups_override"],
        partition_mode=partition_mode if partition_enabled else "plain",
        tau=tau,
        encoder_self_weight=encoder_self_weight,
        encoder_topology_mode=encoder_mode,
        topology_fusion_enabled=bool(extra.get("topology_fusion_enabled", False)),
        fusion_alpha=fusion_alpha,
        fusion_feature_layout=feature_layout,
        use_smoothing_residual=use_smoothing_residual,
        normalize_fusion_feature=normalize_feature,
        batch_size=common["batch_size"],
        device=device,
        cache_dir=None,
    )

    server_train = build_runtime_server_from_affine_context(
        cfg=cfg,
        ctx={
            **ctx,
            "bus_groups": fusion_ctx.bus_groups,
        },
        dataset_for_clients=fusion_ctx.dataset_train_affine,
        feature_mode="topology_local_fusion",
    )
    server_test = build_runtime_server_from_affine_context(
        cfg=cfg,
        ctx={
            **ctx,
            "bus_groups": fusion_ctx.bus_groups,
        },
        dataset_for_clients=fusion_ctx.dataset_test_affine,
        feature_mode="topology_local_fusion",
    )

    return {
        "server_train": server_train,
        "server_test": server_test,
        "dataset_train_affine": fusion_ctx.dataset_train_affine,
        "dataset_test_affine": fusion_ctx.dataset_test_affine,
        "bus_groups": fusion_ctx.bus_groups,
        "bus_groups_string": fusion_ctx.bus_groups_string,
        "extra_tables": {
            "encoder_upload_summary": fusion_ctx.encoder_upload_summary,
            "client_embedding_summary": fusion_ctx.client_embedding_summary,
            "fusion_feature_summary": fusion_ctx.fusion_feature_summary,
        },
        "runtime_topology_info": {
            "bus_axis": int(fusion_ctx.bus_axis),
            "fusion_feature_dim": int(fusion_ctx.dataset_train_affine.feature.shape[1]),
            "bus_groups": fusion_ctx.bus_groups_string,
            "server_feature_layout": feature_layout,
        },
    }
