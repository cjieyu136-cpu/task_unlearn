"""
run_fed_block_tamu_repair.py

Stage 3F-C: client-local Hessian / block-score Fed-TA-MU repair prototype.

This runner should be used only after run_fed_block_influence_audit.py shows that
block-local IHVP and block-local sample scores align with the repo-style global
influence baseline.

What this runner changes relative to Stage 3E
---------------------------------------------
Stage 3E:
    local theta_k VJP for cost test gradient
    repo-style global IHVP / sample score / repair

Stage 3F-C:
    local theta_k / bus-block cost gradient
    local block Hessian H_k
    local block IHVP M_k = -H_k^{-1} g_k
    block sample score score_i = Σ_k ∇theta_k l_{i,k}^T M_k
    repo-style eps solve and repo-style repair update for controlled comparison

Important boundary
------------------
This is still a prototype.

It does NOT yet implement a fully independent client-local parameter update
without assembling back into the global affine-head model. The final repair
evaluation still uses the repo-compatible model reconstruction path so that the
comparison to centralized TA-MU is controlled.

Example:
python run_fed_block_tamu_repair.py model=conv unlearn_prop=0.2 +index_mode=helpful +index_criteria=cost +rho=0.001 +num_bus_clients=4
"""

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

from utils.fed_block_influence import (
    cost_gradient_canonical_by_bus,
    canonical_vec_to_model_order,
    block_influence_audit,
)


def format_float_for_path(x):
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


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


def get_result_dir(cfg, model_type, index_mode, index_criteria, unlearn_prop, rho, num_bus_clients):
    short_name = (
        f"{str(index_mode).lower()}_{index_criteria}"
        f"_p{format_float_for_path(unlearn_prop)}"
        f"_r{format_float_for_path(rho)}"
        f"_bc{int(num_bus_clients)}"
    )
    return Path(str(cfg.simulation_dir)) / model_type / "top_fedtamu" / "stage3f_block_influence" / "block_repair" / short_name


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("========== Stage 3F Block Fed-TA-MU-cost Repair Prototype ==========")

    model_type = str(cfg.model.type)
    if "nn" not in model_type:
        raise ValueError("run_fed_block_tamu_repair.py currently supports nn affine-head setting only.")

    index_mode = OmegaConf.select(cfg, "index_mode", default=cfg.unlearn_mode)
    index_criteria = str(OmegaConf.select(cfg, "index_criteria", default=cfg.criteria))
    selection_policy = OmegaConf.select(cfg, "selection_policy", default=None)
    candidate_ratio = OmegaConf.select(cfg, "candidate_ratio", default=None)

    rho = float(OmegaConf.select(cfg, "rho", default=1e-3))
    damping = float(OmegaConf.select(cfg, "damping", default=1e-8))
    block_damping = float(OmegaConf.select(cfg, "block_damping", default=1e-8))
    train_loss = str(OmegaConf.select(cfg, "train_loss", default="mse"))
    linf_constraint = float(OmegaConf.select(cfg, "linf_constraint", default=1.0))

    batch_size = int(OmegaConf.select(cfg, "fed_block_batch_size", default=128))
    num_bus_clients = int(OmegaConf.select(cfg, "num_bus_clients", default=4))
    bus_groups = OmegaConf.select(cfg, "bus_groups", default=None)

    l1_constraints = get_l1_constraints(cfg)

    print("model_type:", model_type)
    print("index_mode:", index_mode)
    print("index_criteria:", index_criteria)
    print("rho:", rho)
    print("block_damping:", block_damping)
    print("num_bus_clients:", num_bus_clients)
    print("l1_constraints:", l1_constraints)

    # ------------------------------------------------------------
    # Data and Stage-2 topology-aware affine model.
    # ------------------------------------------------------------
    dataset_train, dataset_test = return_dataset(cfg)
    dataset_core, dataset_sensitive = return_core_datasets(cfg, dataset_to_be_split=dataset_train)

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
    print("num unlearn rows:", len(dataset_unlearn))
    print("num remain rows:", len(dataset_remain))

    result_dir = get_result_dir(
        cfg=cfg,
        model_type=model_type,
        index_mode=index_mode,
        index_criteria=index_criteria,
        unlearn_prop=float(cfg.unlearn_prop),
        rho=rho,
        num_bus_clients=num_bus_clients,
    ).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    print("result_dir:", result_dir)
    save_unlearn_object(unlearn_obj, str(result_dir))

    # ------------------------------------------------------------
    # Global repo-style baseline influence.
    # ------------------------------------------------------------
    loader_train = DataLoader(dataset_train_affine, batch_size=batch_size, shuffle=False)
    loader_remain = DataLoader(dataset_remain, batch_size=batch_size, shuffle=False)

    grad_global = centralized_cost_gradient_repo_style(
        cfg=cfg,
        model=model_ori,
        dataset=dataset_test_affine,
        batch_size=batch_size,
    )

    M_global, hvp_info_global = compute_inverse_hvp_vector(
        cfg=cfg,
        model_train=model_ori,
        loader_train=loader_remain,
        vec=grad_global,
        train_loss=train_loss,
        sign=-1.0,
    )

    scores_global, score_info_global = compute_sample_scores(
        cfg=cfg,
        model_train=model_ori,
        loader_hessian=loader_remain,
        loader_score=loader_train,
        dataset_score=dataset_train_affine,
        M_vec=M_global,
        train_loss=train_loss,
        normalize_by_n=True,
    )

    # ------------------------------------------------------------
    # Block-local influence.
    # ------------------------------------------------------------
    audit = block_influence_audit(
        cfg=cfg,
        model=model_ori,
        dataset_test=dataset_test_affine,
        dataset_remain=dataset_remain,
        dataset_score=dataset_train_affine,
        M_global_model_order=M_global,
        scores_global=scores_global,
        num_clients=num_bus_clients,
        bus_groups=bus_groups,
        batch_size=batch_size,
        block_damping=block_damping,
    )

    feature_dim = int(dataset_train_affine.feature.shape[1])
    output_dim = int(dataset_train_affine.target.shape[1])
    grad_block_model_order = canonical_vec_to_model_order(
        model=model_ori,
        feature_dim=feature_dim,
        output_dim=output_dim,
        canonical_vec=audit["grad_result"]["grad_canonical"],
    )

    grad_alignment = alignment_metrics(grad_global, grad_block_model_order)
    M_alignment = audit["M_alignment"]
    score_alignment = audit["score_alignment"]

    print("\n---------- block influence alignment ----------")
    print("grad alignment:", grad_alignment)
    print("M alignment:", M_alignment)
    print("score alignment:", score_alignment)

    scores_block = audit["scores_block"]["scores"]

    np.save(result_dir / "grad_global.npy", grad_global)
    np.save(result_dir / "grad_block_model_order.npy", grad_block_model_order)
    np.save(result_dir / "M_global.npy", M_global)
    np.save(result_dir / "M_block_model_order.npy", audit["M_block_model_order"])
    np.save(result_dir / "scores_global.npy", scores_global)
    np.save(result_dir / "scores_block.npy", scores_block)

    pd.DataFrame(audit["grad_result"]["per_client"]).to_csv(result_dir / "block_grad_client_summary.csv", index=False)
    pd.DataFrame(audit["M_block"]["per_client"]).to_csv(result_dir / "block_M_client_summary.csv", index=False)
    pd.DataFrame(audit["scores_block"]["per_client"]).to_csv(result_dir / "block_score_client_summary.csv", index=False)

    # ------------------------------------------------------------
    # Complete unlearning baseline.
    # ------------------------------------------------------------
    dataset_collection = {
        "remain": dataset_remain,
        "unlearn": dataset_unlearn,
        "test": dataset_test_affine,
    }

    parameter_complete, model_complete, complete_info = repo_style_complete_unlearning(
        cfg=cfg,
        model_original=model_ori,
        loader_remain=loader_remain,
        dataset_remain=dataset_remain,
        parameter_original=parameter_ori,
        train_loss=train_loss,
    )

    metrics_original = evaluate_all(model_ori, dataset_collection, cfg)
    metrics_complete = evaluate_all(model_complete, dataset_collection, cfg)

    # ------------------------------------------------------------
    # Repair grid.
    # ------------------------------------------------------------
    base_info = {
        "model_type": model_type,
        "index_mode": str(index_mode),
        "index_criteria": str(index_criteria),
        "repair_criteria": "cost",
        "selection_policy": str(effective_policy),
        "unlearn_prop": float(cfg.unlearn_prop),
        "rho": float(rho),
        "block_damping": float(block_damping),
        "linf_constraint": float(linf_constraint),
        "num_bus_clients": int(num_bus_clients),
        "bus_groups": audit["info"]["bus_groups"],
        "grad_cosine_similarity": grad_alignment["cosine_similarity"],
        "grad_relative_l2_error": grad_alignment["relative_l2_error"],
        "M_cosine_similarity": M_alignment["cosine_similarity"],
        "M_relative_l2_error": M_alignment["relative_l2_error"],
        "score_cosine_similarity": score_alignment["cosine_similarity"],
        "score_relative_l2_error": score_alignment["relative_l2_error"],
        "H_local_condition": audit["M_block"]["info"]["H_condition"],
        "H_local_shape": str(audit["M_block"]["info"]["H_shape"]),
    }

    rows = [
        build_row("original", None, metrics_original, base_info),
        build_row("complete", None, metrics_complete, base_info),
    ]

    remain_index = unlearn_obj["remain_index"]
    scores_remain_global = scores_global[remain_index]
    scores_remain_block = scores_block[remain_index]

    repair_logs = []

    for l1_constraint in l1_constraints:
        print("----------------------------------------------------")
        print("l1_constraint:", l1_constraint)

        eps_global, eps_info_global = solve_reweight_problem(
            scores_remain=scores_remain_global,
            l1_constraint=float(l1_constraint),
            linf_constraint=linf_constraint,
        )

        parameter_repair_global, model_repair_global, repair_info_global = repo_style_repair_from_eps(
            cfg=cfg,
            model_original=model_ori,
            dataset_remain=dataset_remain,
            eps_remain=eps_global,
            parameter_original=parameter_ori,
            batch_size=batch_size,
            train_loss=train_loss,
        )

        metrics_global = evaluate_all(model_repair_global, dataset_collection, cfg)

        eps_block, eps_info_block = solve_reweight_problem(
            scores_remain=scores_remain_block,
            l1_constraint=float(l1_constraint),
            linf_constraint=linf_constraint,
        )

        parameter_repair_block, model_repair_block, repair_info_block = repo_style_repair_from_eps(
            cfg=cfg,
            model_original=model_ori,
            dataset_remain=dataset_remain,
            eps_remain=eps_block,
            parameter_original=parameter_ori,
            batch_size=batch_size,
            train_loss=train_loss,
        )

        metrics_block = evaluate_all(model_repair_block, dataset_collection, cfg)

        param_diff = float(np.linalg.norm(parameter_repair_block - parameter_repair_global))
        eps_diff = float(np.linalg.norm(eps_block - eps_global))

        print("repo/global metrics:", metrics_global)
        print("block metrics:", metrics_block)
        print("param_diff:", param_diff, "eps_diff:", eps_diff)

        rows.append(
            build_row(
                "repair_global_cost",
                l1_constraint,
                metrics_global,
                base_info,
                eps_info_global,
                repair_info_global,
                {
                    "parameter_l2_diff_to_global_repair": 0.0,
                    "eps_l2_diff_to_global": 0.0,
                },
            )
        )

        rows.append(
            build_row(
                "repair_block_cost",
                l1_constraint,
                metrics_block,
                base_info,
                eps_info_block,
                repair_info_block,
                {
                    "parameter_l2_diff_to_global_repair": param_diff,
                    "eps_l2_diff_to_global": eps_diff,
                    "delta_metric_cost_test_block_minus_global": metrics_block["cost"]["test"] - metrics_global["cost"]["test"],
                    "delta_metric_mse_test_block_minus_global": metrics_block["mse"]["test"] - metrics_global["mse"]["test"],
                    "delta_metric_mape_test_block_minus_global": metrics_block["mape"]["test"] - metrics_global["mape"]["test"],
                    "delta_metric_cost_unlearn_block_minus_global": metrics_block["cost"]["unlearn"] - metrics_global["cost"]["unlearn"],
                },
            )
        )

        repair_logs.append({
            "l1_constraint": float(l1_constraint),
            "global_metrics": metrics_global,
            "block_metrics": metrics_block,
            "global_eps_info": eps_info_global,
            "block_eps_info": eps_info_block,
            "parameter_l2_diff_block_to_global": param_diff,
            "eps_l2_diff_block_to_global": eps_diff,
        })

    summary_df = pd.DataFrame(rows)
    summary_path = result_dir / "fed_block_tamu_repair_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    log = {
        "base_info": base_info,
        "complete_info": complete_info,
        "grad_alignment": grad_alignment,
        "M_alignment": M_alignment,
        "score_alignment": score_alignment,
        "global_hvp_info": hvp_info_global,
        "global_score_info": score_info_global,
        "block_info": audit["info"],
        "M_block_info": audit["M_block"]["info"],
        "repair_logs": repair_logs,
    }
    np.save(result_dir / "fed_block_tamu_repair_log.npy", log, allow_pickle=True)

    print("\n========== Done ==========")
    print("summary:", summary_path)

    show_cols = [
        "method",
        "l1_constraint",
        "metric_mse_test",
        "metric_mape_test",
        "metric_cost_test",
        "eps_linf",
        "eps_min",
        "eps_max",
        "parameter_l2_diff_to_global_repair",
        "eps_l2_diff_to_global",
        "delta_metric_cost_test_block_minus_global",
    ]
    show_cols = [c for c in show_cols if c in summary_df.columns]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
