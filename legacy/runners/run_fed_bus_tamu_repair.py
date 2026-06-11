"""
run_fed_bus_tamu_repair.py

Stage 3D: bus/region-level Fed-TA-MU-cost repair test.

Purpose
-------
This runner extends the bus-group Fed-VJP audit into the TA-MU-cost repair
chain.

It verifies whether a bus/region-client federation can replace the centralized
TA-MU-cost test gradient while preserving repair behavior.

Client definition
-----------------
Each client owns a group of output buses, not a time-sample shard.

Default IEEE-14 split with 4 clients is produced by np.array_split:

    client 0: buses 0,1,2,3
    client 1: buses 4,5,6,7
    client 2: buses 8,9,10
    client 3: buses 11,12,13

CloudServer:
    - computes full OPF/cost-layer gradient:
          g_y = d L_cost / d y_hat
    - sends only g_y[:, bus_group_k] to client k

BusClient k:
    - computes local output VJP:
          grad_k = J_{f[:, bus_group_k]}(theta)^T g_y[:, bus_group_k]

Server:
    - aggregates grad_bus = sum_k grad_k
    - runs the existing repo-style IHVP / sample-score / eps / repair pipeline

Important boundary
------------------
This is the first bus-level repair runner with shared global theta.

It does NOT yet implement independent theta_k per client. That is a separate
client-local-head prototype because it changes the original model parameterization.

To preserve the Stage-2 repo-aligned TA-MU behavior, this runner uses:
    - bus-group Fed-VJP for the TA-MU test gradient
    - existing repo-style compute_sample_scores for sample scores

This is intentional. Hand-writing bus-local train-loss sample scores would risk
diverging from the original repository's influence pipeline. After this passes,
a separate client-local score decomposition can be added and tested against this
repo-style score baseline.

Example
-------
python run_fed_bus_tamu_repair.py model=conv unlearn_prop=0.2 +index_mode=helpful +index_criteria=cost +rho=0.001 +num_bus_clients=4

Optional explicit groups:
python run_fed_bus_tamu_repair.py model=conv unlearn_prop=0.2 +index_mode=helpful +index_criteria=cost +rho=0.001 '+bus_groups=0,1,2,3|4,5,6,7|8,9,10|11,12,13'
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import hydra

from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from utils import return_dataset, evaluate
from utils.optimization import Operator
from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.topo_affine import return_topology_affine_model

from func_operation import (
    return_core_datasets,
    return_dataset_for_nn_affine,
)

from utils.index_utils import (
    load_and_split_unlearn_datasets,
    save_unlearn_object,
)

from utils.reweight_utils import (
    get_model_parameter_vector,
    compute_inverse_hvp_vector,
    compute_sample_scores,
    solve_reweight_problem,
    repo_style_complete_unlearning,
    repo_style_repair_from_eps,
)

from utils.fed_vjp_utils import (
    centralized_cost_gradient_repo_style,
    alignment_metrics,
)

from utils.fed_bus_client import (
    compare_bus_fed_to_central_gradient,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def format_float_for_path(x):
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def safe_tag(x):
    return (
        str(x)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(".", "p")
        .replace("|", "-")
        .replace(",", "")
        .replace(" ", "")
    )


def get_l1_constraints(cfg):
    root = OmegaConf.select(cfg, "l1_constraints", default=None)
    if root is not None:
        if isinstance(root, (float, int)):
            return [float(root)]
        return [float(v) for v in list(root)]

    model_level = OmegaConf.select(cfg, "model.l1_constraints", default=None)
    if model_level is not None:
        if isinstance(model_level, (float, int)):
            return [float(model_level)]
        return [float(v) for v in list(model_level)]

    return [0.15, 0.125, 0.1, 0.075, 0.05, 0.025, 0.0]


def evaluate_all(model, dataset_collection, cfg):
    return {
        "mse": evaluate(model, dataset_collection, loss="mse", case_config=cfg.case),
        "mape": evaluate(model, dataset_collection, loss="mape", case_config=cfg.case),
        "cost": evaluate(model, dataset_collection, loss="cost", case_config=cfg.case),
    }


def flatten_metrics(prefix, metrics):
    row = {}
    for loss_name, split_dict in metrics.items():
        for split_name, value in split_dict.items():
            row[f"{prefix}_{loss_name}_{split_name}"] = float(value)
    return row


def build_row(method, l1_constraint, metrics, base_info, eps_info=None, repair_info=None, diff_info=None):
    row = dict(base_info)
    row["method"] = method
    row["l1_constraint"] = np.nan if l1_constraint is None else float(l1_constraint)

    row.update(flatten_metrics("metric", metrics))

    if eps_info:
        for k in ["eps_l1", "eps_linf", "eps_min", "eps_max", "solver", "status"]:
            row[k] = eps_info.get(k, np.nan)

    if repair_info:
        row["weighted_ihvp_norm"] = repair_info.get("weighted_ihvp_norm", np.nan)
        row["parameter_repair_norm"] = repair_info.get("parameter_repair_norm", np.nan)

    if diff_info:
        row.update(diff_info)

    return row


def get_result_dir(
    cfg,
    model_type,
    index_mode,
    index_criteria,
    selection_policy,
    unlearn_prop,
    rho,
    linf,
    num_bus_clients,
    bus_groups_string,
):
    """
    Keep this directory deliberately short.

    Windows often fails on long nested result paths, and pandas.to_csv can raise
    FileNotFoundError even when the parent mkdir line is present. We store full
    metadata inside the CSV/log instead of the directory name.
    """
    short_name = (
        f"{str(index_mode).lower()}_{index_criteria}"
        f"_p{format_float_for_path(unlearn_prop)}"
        f"_r{format_float_for_path(rho)}"
        f"_bc{int(num_bus_clients)}"
    )

    return os.path.join(
        str(cfg.simulation_dir),
        model_type,
        "top_fedtamu",
        "stage3d_fed_bus_tamu",
        "fed_bus_tamu_repair",
        short_name,
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("========== Bus-group Fed-TA-MU-cost Repair Test ==========")

    model_type = str(cfg.model.type)
    if "nn" not in model_type:
        raise ValueError("run_fed_bus_tamu_repair.py currently supports nn affine-head setting only.")

    index_mode = OmegaConf.select(cfg, "index_mode", default=cfg.unlearn_mode)
    index_criteria = str(OmegaConf.select(cfg, "index_criteria", default=cfg.criteria))
    selection_policy = OmegaConf.select(cfg, "selection_policy", default=None)
    candidate_ratio = OmegaConf.select(cfg, "candidate_ratio", default=None)

    rho = float(OmegaConf.select(cfg, "rho", default=1e-3))
    damping = float(OmegaConf.select(cfg, "damping", default=1e-8))
    linf_constraint = float(OmegaConf.select(cfg, "linf_constraint", default=1.0))
    train_loss = str(OmegaConf.select(cfg, "train_loss", default="mse"))

    batch_size = int(OmegaConf.select(cfg, "fed_bus_batch_size", default=128))
    num_bus_clients = int(OmegaConf.select(cfg, "num_bus_clients", default=4))
    bus_groups = OmegaConf.select(cfg, "bus_groups", default=None)

    l1_constraints = get_l1_constraints(cfg)

    print("model_type:", model_type)
    print("index_mode:", index_mode)
    print("index_criteria:", index_criteria)
    print("selection_policy override:", selection_policy)
    print("rho:", rho)
    print("damping:", damping)
    print("linf_constraint:", linf_constraint)
    print("batch_size:", batch_size)
    print("num_bus_clients:", num_bus_clients)
    print("bus_groups override:", bus_groups)
    print("l1_constraints:", l1_constraints)

    # ------------------------------------------------------------
    # Data and Stage-2 topology-aware affine model.
    # ------------------------------------------------------------
    dataset_train, dataset_test = return_dataset(cfg)

    dataset_core, dataset_sensitive = return_core_datasets(
        cfg,
        dataset_to_be_split=dataset_train,
    )

    dataset_train_affine, dataset_test_affine = return_dataset_for_nn_affine(
        cfg,
        dataset_sensitive,
        dataset_test,
    )

    print("train affine feature shape:", dataset_train_affine.feature.shape)
    print("test affine feature shape:", dataset_test_affine.feature.shape)

    operator = Operator(case_config=cfg.case)
    A_grid = build_load_bus_adjacency(operator)
    L_grid = build_laplacian(A_grid)

    model_ori, parameter_ori_raw = return_topology_affine_model(
        dataset=dataset_train_affine,
        L_grid=L_grid,
        rho=rho,
        damping=damping,
    )
    model_ori.eval()

    parameter_ori = get_model_parameter_vector(model_ori)
    parameter_ori_raw = np.asarray(parameter_ori_raw, dtype=float).reshape(-1)
    raw_diff = float(np.linalg.norm(parameter_ori - parameter_ori_raw))

    print("flatten_model parameter norm:", float(np.linalg.norm(parameter_ori)))
    print("raw parameter norm:", float(np.linalg.norm(parameter_ori_raw)))
    print("norm(flatten_model - raw_parameter):", raw_diff)
    if raw_diff > 1e-4:
        print("[WARN] raw parameter differs from flatten_model(model_ori). Using flatten_model for repo-style update.")

    # ------------------------------------------------------------
    # Index split.
    # ------------------------------------------------------------
    dataset_unlearn, dataset_remain, unlearn_obj = load_and_split_unlearn_datasets(
        cfg=cfg,
        model_type=model_type,
        dataset=dataset_train_affine,
        index_mode=index_mode,
        index_criteria=index_criteria,
        unlearn_prop=float(cfg.unlearn_prop),
        selection_policy=selection_policy,
        candidate_ratio=candidate_ratio,
    )

    effective_policy = unlearn_obj["metadata"].get("selection_policy", "event_or_none")

    print("---------- index metadata ----------")
    print("selection_policy:", effective_policy)
    print("candidate_ratio:", unlearn_obj["metadata"].get("candidate_ratio"))
    print("candidate_no:", unlearn_obj["metadata"].get("candidate_no"))
    print("unlearn_no:", unlearn_obj["metadata"].get("unlearn_no"))
    print("selected_score_sum:", unlearn_obj["metadata"].get("selected_score_sum"))
    print("num unlearn rows:", len(dataset_unlearn))
    print("num remain rows:", len(dataset_remain))
    print("first 20 unlearn_index:", unlearn_obj["unlearn_index"][:20])

    # ------------------------------------------------------------
    # Bus-group Fed-VJP gradient on test set.
    # ------------------------------------------------------------
    print("\n---------- bus-group Fed-VJP gradient ----------")

    bus_grad_result = compare_bus_fed_to_central_gradient(
        cfg=cfg,
        model=model_ori,
        dataset=dataset_test_affine,
        num_clients=num_bus_clients,
        bus_groups=bus_groups,
        batch_size=batch_size,
    )

    central_grad = bus_grad_result["grad_centralized"]
    fed_bus_grad = bus_grad_result["grad_fed_bus"]
    grad_alignment = bus_grad_result["alignment"]
    bus_groups_string = bus_grad_result["bus_groups_string"]

    print("bus groups:", bus_groups_string)
    print("gradient alignment:", grad_alignment)
    print("bus client info:")
    for item in bus_grad_result["per_client"]:
        print(item)

    result_dir = get_result_dir(
        cfg=cfg,
        model_type=model_type,
        index_mode=index_mode,
        index_criteria=index_criteria,
        selection_policy=effective_policy,
        unlearn_prop=float(cfg.unlearn_prop),
        rho=rho,
        linf=linf_constraint,
        num_bus_clients=num_bus_clients,
        bus_groups_string=bus_groups_string,
    )
    # Use an absolute resolved path to avoid Windows/Hydra relative-path issues.
    result_dir = str(Path(result_dir).resolve())
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    print("result_dir:", result_dir)

    save_unlearn_object(unlearn_obj, result_dir)

    # Defensive file writes. Create parent dirs immediately before each write.
    payload_path = Path(result_dir) / "fed_bus_client_payload.csv"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(bus_grad_result["per_client"]).to_csv(payload_path, index=False)

    grad_central_path = Path(result_dir) / "grad_centralized_cost.npy"
    grad_fed_path = Path(result_dir) / "grad_fed_bus_cost.npy"
    gy_path = Path(result_dir) / "gy_test_scaled.npy"
    grad_central_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(grad_central_path, central_grad)
    np.save(grad_fed_path, fed_bus_grad)
    np.save(gy_path, bus_grad_result["gy_scaled"])

    # ------------------------------------------------------------
    # Loaders.
    # ------------------------------------------------------------
    loader_train = DataLoader(dataset_train_affine, batch_size=batch_size, shuffle=False)
    loader_remain = DataLoader(dataset_remain, batch_size=batch_size, shuffle=False)

    dataset_collection = {
        "remain": dataset_remain,
        "unlearn": dataset_unlearn,
        "test": dataset_test_affine,
    }

    # ------------------------------------------------------------
    # Complete unlearning.
    # ------------------------------------------------------------
    parameter_complete, model_complete, complete_info = repo_style_complete_unlearning(
        cfg=cfg,
        model_original=model_ori,
        loader_remain=loader_remain,
        dataset_remain=dataset_remain,
        parameter_original=parameter_ori,
        train_loss=train_loss,
    )

    # ------------------------------------------------------------
    # IHVP directions.
    # ------------------------------------------------------------
    M_central, hvp_info_central = compute_inverse_hvp_vector(
        cfg=cfg,
        model_train=model_ori,
        loader_train=loader_remain,
        vec=central_grad,
        train_loss=train_loss,
        sign=-1.0,
    )

    M_fed, hvp_info_fed = compute_inverse_hvp_vector(
        cfg=cfg,
        model_train=model_ori,
        loader_train=loader_remain,
        vec=fed_bus_grad,
        train_loss=train_loss,
        sign=-1.0,
    )

    M_alignment = alignment_metrics(M_central, M_fed)
    print("M alignment:", M_alignment)

    # ------------------------------------------------------------
    # Scores.
    # Keep this repo-style and centralized for this first repair runner.
    # This prevents hand-written score decomposition from diverging from the
    # already-validated original TA-MU influence pipeline.
    # ------------------------------------------------------------
    scores_central, score_info_central = compute_sample_scores(
        cfg=cfg,
        model_train=model_ori,
        loader_hessian=loader_remain,
        loader_score=loader_train,
        dataset_score=dataset_train_affine,
        M_vec=M_central,
        train_loss=train_loss,
        normalize_by_n=True,
    )

    scores_fed, score_info_fed = compute_sample_scores(
        cfg=cfg,
        model_train=model_ori,
        loader_hessian=loader_remain,
        loader_score=loader_train,
        dataset_score=dataset_train_affine,
        M_vec=M_fed,
        train_loss=train_loss,
        normalize_by_n=True,
    )

    score_alignment = alignment_metrics(scores_central, scores_fed)
    print("score alignment:", score_alignment)

    m_central_path = Path(result_dir) / "M_central.npy"
    m_central_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(m_central_path, M_central)
    np.save(Path(result_dir) / "M_fed_bus.npy", M_fed)
    np.save(Path(result_dir) / "scores_central.npy", scores_central)
    np.save(Path(result_dir) / "scores_fed_bus.npy", scores_fed)

    # ------------------------------------------------------------
    # Metrics.
    # ------------------------------------------------------------
    metrics_original = evaluate_all(model_ori, dataset_collection, cfg)
    metrics_complete = evaluate_all(model_complete, dataset_collection, cfg)

    base_info = {
        "model_type": model_type,
        "index_mode": str(index_mode),
        "index_criteria": str(index_criteria),
        "repair_criteria": "cost",
        "selection_policy": str(effective_policy),
        "unlearn_prop": float(cfg.unlearn_prop),
        "rho": float(rho),
        "linf_constraint": float(linf_constraint),
        "num_bus_clients": int(num_bus_clients),
        "bus_groups": bus_groups_string,
        "grad_cosine_similarity": grad_alignment["cosine_similarity"],
        "grad_relative_l2_error": grad_alignment["relative_l2_error"],
        "M_cosine_similarity": M_alignment["cosine_similarity"],
        "M_relative_l2_error": M_alignment["relative_l2_error"],
        "score_cosine_similarity": score_alignment["cosine_similarity"],
        "score_relative_l2_error": score_alignment["relative_l2_error"],
    }

    rows = [
        build_row("original", None, metrics_original, base_info),
        build_row("complete", None, metrics_complete, base_info),
    ]

    remain_index = unlearn_obj["remain_index"]
    scores_remain_central = scores_central[remain_index]
    scores_remain_fed = scores_fed[remain_index]

    repair_logs = []

    # ------------------------------------------------------------
    # Repair grid.
    # ------------------------------------------------------------
    for l1_constraint in l1_constraints:
        print("----------------------------------------------------")
        print("l1_constraint:", l1_constraint)

        eps_central, eps_info_central = solve_reweight_problem(
            scores_remain=scores_remain_central,
            l1_constraint=float(l1_constraint),
            linf_constraint=linf_constraint,
        )

        parameter_repair_central, model_repair_central, repair_info_central = repo_style_repair_from_eps(
            cfg=cfg,
            model_original=model_ori,
            dataset_remain=dataset_remain,
            eps_remain=eps_central,
            parameter_original=parameter_ori,
            batch_size=batch_size,
            train_loss=train_loss,
        )

        metrics_central = evaluate_all(model_repair_central, dataset_collection, cfg)

        eps_fed, eps_info_fed = solve_reweight_problem(
            scores_remain=scores_remain_fed,
            l1_constraint=float(l1_constraint),
            linf_constraint=linf_constraint,
        )

        parameter_repair_fed, model_repair_fed, repair_info_fed = repo_style_repair_from_eps(
            cfg=cfg,
            model_original=model_ori,
            dataset_remain=dataset_remain,
            eps_remain=eps_fed,
            parameter_original=parameter_ori,
            batch_size=batch_size,
            train_loss=train_loss,
        )

        metrics_fed = evaluate_all(model_repair_fed, dataset_collection, cfg)

        param_diff = float(np.linalg.norm(parameter_repair_fed - parameter_repair_central))
        eps_diff = float(np.linalg.norm(eps_fed - eps_central))

        print("centralized metrics:", metrics_central)
        print("bus-fed metrics:", metrics_fed)
        print("param_diff:", param_diff, "eps_diff:", eps_diff)

        diff_info_central = {
            "parameter_l2_diff_to_central_repair": 0.0,
            "eps_l2_diff_to_central": 0.0,
        }

        diff_info_fed = {
            "parameter_l2_diff_to_central_repair": param_diff,
            "eps_l2_diff_to_central": eps_diff,
            "delta_metric_cost_test_fed_minus_central": metrics_fed["cost"]["test"] - metrics_central["cost"]["test"],
            "delta_metric_mse_test_fed_minus_central": metrics_fed["mse"]["test"] - metrics_central["mse"]["test"],
            "delta_metric_mape_test_fed_minus_central": metrics_fed["mape"]["test"] - metrics_central["mape"]["test"],
            "delta_metric_cost_unlearn_fed_minus_central": metrics_fed["cost"]["unlearn"] - metrics_central["cost"]["unlearn"],
        }

        rows.append(
            build_row(
                "repair_central_cost",
                l1_constraint,
                metrics_central,
                base_info,
                eps_info_central,
                repair_info_central,
                diff_info_central,
            )
        )

        rows.append(
            build_row(
                "repair_bus_fed_cost",
                l1_constraint,
                metrics_fed,
                base_info,
                eps_info_fed,
                repair_info_fed,
                diff_info_fed,
            )
        )

        repair_logs.append(
            {
                "l1_constraint": float(l1_constraint),
                "central_metrics": metrics_central,
                "bus_fed_metrics": metrics_fed,
                "central_eps_info": eps_info_central,
                "bus_fed_eps_info": eps_info_fed,
                "parameter_l2_diff_fed_to_central": param_diff,
                "eps_l2_diff_fed_to_central": eps_diff,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_path = Path(result_dir) / "fed_bus_tamu_repair_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    log = {
        "base_info": base_info,
        "complete_info": complete_info,
        "grad_alignment": grad_alignment,
        "M_alignment": M_alignment,
        "score_alignment": score_alignment,
        "hvp_info_central": hvp_info_central,
        "hvp_info_fed": hvp_info_fed,
        "score_info_central": score_info_central,
        "score_info_fed": score_info_fed,
        "bus_grad_info": bus_grad_result["fed_info"],
        "bus_grad_per_client": bus_grad_result["per_client"],
        "repair_logs": repair_logs,
    }
    log_path = Path(result_dir) / "fed_bus_tamu_repair_log.npy"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(log_path, log, allow_pickle=True)

    print("\n========== Done ==========")
    print("summary:", summary_path)
    print("client payload:", Path(result_dir) / "fed_bus_client_payload.csv")

    show_cols = [
        "method",
        "l1_constraint",
        "metric_mse_test",
        "metric_mape_test",
        "metric_cost_test",
        "eps_linf",
        "eps_min",
        "eps_max",
        "parameter_l2_diff_to_central_repair",
        "eps_l2_diff_to_central",
        "delta_metric_cost_test_fed_minus_central",
    ]
    show_cols = [c for c in show_cols if c in summary_df.columns]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
