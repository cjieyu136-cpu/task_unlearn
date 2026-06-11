"""
run_fed_local_feature_audit.py

Stage 3H H2-2 local frozen-backbone feature audit.

Purpose
-------
Check whether a FedClient can generate the same affine features locally by
running the frozen core NN backbone as the repository's centralized
return_dataset_for_nn_affine(...) path.

This is the first step toward moving from:

    centralized affine feature matrix
    -> client feature cache simulation

to:

    client raw/local data
    -> client local frozen backbone
    -> local feature h_k(x)

Important boundary
------------------
This script does NOT yet implement secure aggregation or multi-machine clients.

It audits the repository-compatible frozen-backbone feature generation. In the
current repo architecture, the core NN expects the same input tensor format used
by the centralized dataset. Therefore this first audit uses:

    feature_mode = local_frozen_backbone_full_input

Meaning:
    each FedClient locally runs the frozen core model on its local copy of the
    input tensor and compares against the repo centralized affine features.

If a stricter industrial setting requires each regional/bus client to own only
partial raw inputs, the core model architecture or input interface must be
changed. That is a separate task.

Example
-------
python run_fed_local_feature_audit.py model=conv +rho=0.001 +num_bus_clients=4
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from utils import return_dataset
from func_operation import (
    return_core_datasets,
    return_dataset_for_nn_affine,
    return_nn_model,
)
from utils.optimization import Operator
from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.topo_affine import return_topology_affine_model
from utils.fed_bus_client import parse_bus_groups, bus_groups_to_string
from utils.fed_local_head import extract_affine_weight_bias


def format_float_for_path(x):
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def feature_alignment(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    diff = a - b
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(np.dot(a, b) / denom) if denom > 0 else np.nan
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "relative_l2_error": float(np.linalg.norm(diff) / (np.linalg.norm(a) + 1e-12)),
        "cosine_similarity": cos,
    }


@torch.no_grad()
def compute_core_features(core_model, raw_feature, batch_size=128, device="cpu"):
    """
    Run frozen core model locally and return feature h(x).

    For NN_CONV and MLPMixer in this repo, forward returns:
        feature, forecast
    """
    core_model = core_model.to(device)
    core_model.eval()

    x = raw_feature
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
    else:
        x = x.float()

    feats = []
    for start in range(0, len(x), int(batch_size)):
        xb = x[start:start + int(batch_size)].to(device)
        out = core_model(xb)
        if isinstance(out, tuple) or isinstance(out, list):
            feat = out[0]
        else:
            raise RuntimeError(
                "core_model forward did not return (feature, forecast). "
                "Please check utils/net.py or the selected model type."
            )
        feats.append(feat.detach().cpu())
    return torch.cat(feats, dim=0)


def make_result_dir(cfg, model_type, rho, num_bus_clients):
    return (
        Path(str(cfg.simulation_dir))
        / model_type
        / "top_fedtamu"
        / "stage3h_runtime"
        / "local_feature_audit"
        / f"rho_{format_float_for_path(rho)}_bc{int(num_bus_clients)}"
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("========== Stage 3H Local Frozen-Backbone Feature Audit ==========")

    model_type = str(cfg.model.type)
    if model_type not in ["nn_conv", "nn_mixer"]:
        raise ValueError(
            "run_fed_local_feature_audit.py currently supports nn_conv / nn_mixer only."
        )

    rho = float(OmegaConf.select(cfg, "rho", default=1e-3))
    damping = float(OmegaConf.select(cfg, "damping", default=1e-8))
    batch_size = int(OmegaConf.select(cfg, "fed_feature_batch_size", default=128))
    num_bus_clients = int(OmegaConf.select(cfg, "num_bus_clients", default=4))
    bus_groups_override = OmegaConf.select(cfg, "bus_groups", default=None)
    device = str(OmegaConf.select(cfg, "device", default="cpu"))

    print("model_type:", model_type)
    print("rho:", rho)
    print("damping:", damping)
    print("num_bus_clients:", num_bus_clients)
    print("feature_mode:", "local_frozen_backbone_full_input")
    print("device:", device)

    # ------------------------------------------------------------------
    # Repository data path.
    # ------------------------------------------------------------------
    dataset_train, dataset_test = return_dataset(cfg)
    dataset_core, dataset_sensitive = return_core_datasets(
        cfg,
        dataset_to_be_split=dataset_train,
    )

    # Centralized repo affine features. This is the reference to match.
    dataset_train_affine, dataset_test_affine = return_dataset_for_nn_affine(
        cfg,
        dataset_sensitive,
        dataset_test,
    )

    print("raw sensitive feature shape:", tuple(dataset_sensitive.feature.shape))
    print("central train affine feature shape:", tuple(dataset_train_affine.feature.shape))
    print("raw test feature shape:", tuple(dataset_test.feature.shape))
    print("central test affine feature shape:", tuple(dataset_test_affine.feature.shape))

    # ------------------------------------------------------------------
    # Local frozen-backbone feature generation.
    # ------------------------------------------------------------------
    core_model = return_nn_model(cfg, is_load=True, dataset="core")
    core_model.eval()

    local_train_feature = compute_core_features(
        core_model=core_model,
        raw_feature=dataset_sensitive.feature,
        batch_size=batch_size,
        device=device,
    )
    local_test_feature = compute_core_features(
        core_model=core_model,
        raw_feature=dataset_test.feature,
        batch_size=batch_size,
        device=device,
    )

    central_train_feature = dataset_train_affine.feature
    central_test_feature = dataset_test_affine.feature

    train_align = feature_alignment(local_train_feature, central_train_feature)
    test_align = feature_alignment(local_test_feature, central_test_feature)

    print("---------- global feature alignment ----------")
    print("train:", train_align)
    print("test:", test_align)

    # ------------------------------------------------------------------
    # Bus-client interpretation.
    # ------------------------------------------------------------------
    output_dim = int(dataset_train_affine.target.shape[1])
    feature_dim = int(dataset_train_affine.feature.shape[1])
    bus_groups = parse_bus_groups(
        bus_groups=bus_groups_override,
        num_bus=output_dim,
        num_clients=num_bus_clients,
    )

    # Optional: train topology-affine head and check local prediction slice
    # equivalence using locally generated features.
    operator = Operator(case_config=cfg.case)
    A_grid = build_load_bus_adjacency(operator)
    L_grid = build_laplacian(A_grid)

    model_affine, _parameter = return_topology_affine_model(
        dataset=dataset_train_affine,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
    )
    model_affine.eval()

    affine = extract_affine_weight_bias(
        model_affine,
        feature_dim=feature_dim,
        output_dim=output_dim,
    )
    W_out_in = affine["W_out_in"]
    b = affine["b"]

    with torch.no_grad():
        pred_central_train = model_affine(dataset_train_affine.feature).detach().cpu().numpy()
        pred_central_test = model_affine(dataset_test_affine.feature).detach().cpu().numpy()

    local_train_np = to_numpy(local_train_feature)
    local_test_np = to_numpy(local_test_feature)

    client_rows = []
    for cid, buses in enumerate(bus_groups):
        buses = np.asarray(buses, dtype=int)

        # Each bus-client locally runs the same frozen backbone on its local
        # input cache, then uses only its local output head rows.
        pred_local_train = local_train_np @ W_out_in[buses, :].T + b[buses].reshape(1, -1)
        pred_local_test = local_test_np @ W_out_in[buses, :].T + b[buses].reshape(1, -1)

        pred_ref_train = pred_central_train[:, buses]
        pred_ref_test = pred_central_test[:, buses]

        pa_train = feature_alignment(pred_local_train, pred_ref_train)
        pa_test = feature_alignment(pred_local_test, pred_ref_test)

        row = {
            "client_id": int(cid),
            "bus_indices": ",".join(str(int(x)) for x in buses),
            "num_buses": int(len(buses)),
            "feature_mode": "local_frozen_backbone_full_input",
            "train_feature_rmse": train_align["rmse"],
            "train_feature_max_abs_diff": train_align["max_abs_diff"],
            "train_feature_cosine_similarity": train_align["cosine_similarity"],
            "test_feature_rmse": test_align["rmse"],
            "test_feature_max_abs_diff": test_align["max_abs_diff"],
            "test_feature_cosine_similarity": test_align["cosine_similarity"],
            "train_prediction_rmse": pa_train["rmse"],
            "train_prediction_max_abs_diff": pa_train["max_abs_diff"],
            "train_prediction_cosine_similarity": pa_train["cosine_similarity"],
            "test_prediction_rmse": pa_test["rmse"],
            "test_prediction_max_abs_diff": pa_test["max_abs_diff"],
            "test_prediction_cosine_similarity": pa_test["cosine_similarity"],
            "theta_k_elements": int(len(buses) * (feature_dim + 1)),
        }
        client_rows.append(row)

    client_df = pd.DataFrame(client_rows)

    summary = {
        "model_type": model_type,
        "rho": rho,
        "damping": damping,
        "num_bus_clients": int(num_bus_clients),
        "bus_groups": bus_groups_to_string(bus_groups),
        "feature_mode": "local_frozen_backbone_full_input",
        "train_num_samples": int(len(dataset_train_affine)),
        "test_num_samples": int(len(dataset_test_affine)),
        "feature_dim": int(feature_dim),
        "output_dim": int(output_dim),
        "train_feature_rmse": train_align["rmse"],
        "train_feature_max_abs_diff": train_align["max_abs_diff"],
        "train_feature_mean_abs_diff": train_align["mean_abs_diff"],
        "train_feature_relative_l2_error": train_align["relative_l2_error"],
        "train_feature_cosine_similarity": train_align["cosine_similarity"],
        "test_feature_rmse": test_align["rmse"],
        "test_feature_max_abs_diff": test_align["max_abs_diff"],
        "test_feature_mean_abs_diff": test_align["mean_abs_diff"],
        "test_feature_relative_l2_error": test_align["relative_l2_error"],
        "test_feature_cosine_similarity": test_align["cosine_similarity"],
        "max_client_train_prediction_rmse": float(client_df["train_prediction_rmse"].max()),
        "max_client_test_prediction_rmse": float(client_df["test_prediction_rmse"].max()),
        "total_theta_k_elements": int(client_df["theta_k_elements"].sum()),
    }

    result_dir = make_result_dir(cfg, model_type, rho, num_bus_clients).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    summary_path = result_dir / "fed_local_feature_audit_summary.csv"
    client_path = result_dir / "fed_local_feature_client_summary.csv"
    log_path = result_dir / "fed_local_feature_audit_log.npy"

    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    client_df.to_csv(client_path, index=False)

    log = {
        "summary": summary,
        "client_rows": client_rows,
        "train_alignment": train_align,
        "test_alignment": test_align,
    }
    np.save(log_path, log, allow_pickle=True)

    print("\n========== Done ==========")
    print("summary:", summary_path)
    print("client summary:", client_path)
    print(client_df.to_string(index=False))


if __name__ == "__main__":
    main()
