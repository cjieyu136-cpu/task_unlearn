"""
run_fed_block_influence_audit.py

Stage 3F-A/B: client-local Hessian / block influence audit.

This runner does NOT perform repair.

It compares repo-style global influence against block-local influence:

    global:
        M_global = -H_global^{-1} g
        score_global_i = ∇θ l_i^T M_global

    block-local:
        M_k = -H_k^{-1} g_k
        score_block_i = Σ_k ∇θ_k l_{i,k}^T M_k

The goal is to determine whether a client-local Hessian/block approximation is
close enough to the repo-style TA-MU baseline to justify a later block repair
prototype.

Example:
python run_fed_block_influence_audit.py model=conv unlearn_prop=0.2 +index_mode=helpful +index_criteria=cost +rho=0.001 +num_bus_clients=4
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import hydra

from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from utils import return_dataset
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


def get_result_dir(cfg, model_type, index_mode, index_criteria, unlearn_prop, rho, num_bus_clients):
    short_name = (
        f"{str(index_mode).lower()}_{index_criteria}"
        f"_p{format_float_for_path(unlearn_prop)}"
        f"_r{format_float_for_path(rho)}"
        f"_bc{int(num_bus_clients)}"
    )
    return Path(str(cfg.simulation_dir)) / model_type / "top_fedtamu" / "stage3f_block_influence" / "block_audit" / short_name


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("========== Stage 3F Block Influence Audit ==========")

    model_type = str(cfg.model.type)
    if "nn" not in model_type:
        raise ValueError("run_fed_block_influence_audit.py currently supports nn affine-head setting only.")

    index_mode = OmegaConf.select(cfg, "index_mode", default=cfg.unlearn_mode)
    index_criteria = str(OmegaConf.select(cfg, "index_criteria", default=cfg.criteria))
    selection_policy = OmegaConf.select(cfg, "selection_policy", default=None)
    candidate_ratio = OmegaConf.select(cfg, "candidate_ratio", default=None)

    rho = float(OmegaConf.select(cfg, "rho", default=1e-3))
    damping = float(OmegaConf.select(cfg, "damping", default=1e-8))
    train_loss = str(OmegaConf.select(cfg, "train_loss", default="mse"))

    batch_size = int(OmegaConf.select(cfg, "fed_block_batch_size", default=128))
    num_bus_clients = int(OmegaConf.select(cfg, "num_bus_clients", default=4))
    bus_groups = OmegaConf.select(cfg, "bus_groups", default=None)
    block_damping = float(OmegaConf.select(cfg, "block_damping", default=1e-8))

    print("model_type:", model_type)
    print("index_mode:", index_mode)
    print("index_criteria:", index_criteria)
    print("rho:", rho)
    print("num_bus_clients:", num_bus_clients)
    print("block_damping:", block_damping)

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
    print("flatten_model parameter norm:", float(np.linalg.norm(parameter_ori)))
    print("raw parameter norm:", float(np.linalg.norm(parameter_ori_raw)))
    print("norm(flatten_model - raw_parameter):", float(np.linalg.norm(parameter_ori - parameter_ori_raw)))

    # ------------------------------------------------------------
    # Index split for remain set / Hessian.
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
    # Global repo-style baseline.
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

    # Compare local cost gradient to global gradient in model order.
    feature_dim = int(dataset_train_affine.feature.shape[1])
    output_dim = int(dataset_train_affine.target.shape[1])
    grad_block_model_order = canonical_vec_to_model_order(
        model=model_ori,
        feature_dim=feature_dim,
        output_dim=output_dim,
        canonical_vec=audit["grad_result"]["grad_canonical"],
    )
    grad_alignment = alignment_metrics(grad_global, grad_block_model_order)

    print("\n---------- alignment ----------")
    print("grad alignment:", grad_alignment)
    print("M alignment:", audit["M_alignment"])
    print("score alignment:", audit["score_alignment"])

    print("\n---------- block Hessian info ----------")
    print(audit["M_block"]["info"])

    print("\n---------- block clients ----------")
    for item in audit["M_block"]["per_client"]:
        print(item)

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------
    np.save(result_dir / "grad_global.npy", grad_global)
    np.save(result_dir / "grad_block_model_order.npy", grad_block_model_order)
    np.save(result_dir / "M_global.npy", M_global)
    np.save(result_dir / "M_block_model_order.npy", audit["M_block_model_order"])
    np.save(result_dir / "M_block_canonical.npy", audit["M_block"]["M_canonical"])
    np.save(result_dir / "scores_global.npy", scores_global)
    np.save(result_dir / "scores_block.npy", audit["scores_block"]["scores"])

    pd.DataFrame(audit["grad_result"]["per_client"]).to_csv(result_dir / "block_grad_client_summary.csv", index=False)
    pd.DataFrame(audit["M_block"]["per_client"]).to_csv(result_dir / "block_M_client_summary.csv", index=False)
    pd.DataFrame(audit["scores_block"]["per_client"]).to_csv(result_dir / "block_score_client_summary.csv", index=False)

    summary = {
        "model_type": model_type,
        "index_mode": str(index_mode),
        "index_criteria": index_criteria,
        "selection_policy": str(effective_policy),
        "unlearn_prop": float(cfg.unlearn_prop),
        "rho": rho,
        "num_bus_clients": int(num_bus_clients),
        "bus_groups": audit["info"]["bus_groups"],
        "block_damping": block_damping,
        "grad_cosine_similarity": grad_alignment["cosine_similarity"],
        "grad_relative_l2_error": grad_alignment["relative_l2_error"],
        "M_cosine_similarity": audit["M_alignment"]["cosine_similarity"],
        "M_relative_l2_error": audit["M_alignment"]["relative_l2_error"],
        "score_cosine_similarity": audit["score_alignment"]["cosine_similarity"],
        "score_relative_l2_error": audit["score_alignment"]["relative_l2_error"],
        "M_global_norm": audit["M_alignment"]["centralized_norm"],
        "M_block_norm": audit["M_alignment"]["fed_vjp_norm"],
        "score_global_norm": audit["score_alignment"]["centralized_norm"],
        "score_block_norm": audit["score_alignment"]["fed_vjp_norm"],
        "H_local_condition": audit["M_block"]["info"]["H_condition"],
        "H_local_shape": str(audit["M_block"]["info"]["H_shape"]),
        "global_hvp_info": str(hvp_info_global),
        "global_score_info": str(score_info_global),
    }

    summary_path = result_dir / "fed_block_influence_audit_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    log = {
        "summary": summary,
        "grad_alignment": grad_alignment,
        "M_alignment": audit["M_alignment"],
        "score_alignment": audit["score_alignment"],
        "block_info": audit["info"],
        "M_block_info": audit["M_block"]["info"],
        "global_hvp_info": hvp_info_global,
        "global_score_info": score_info_global,
    }
    np.save(result_dir / "fed_block_influence_audit_log.npy", log, allow_pickle=True)

    print("\n========== Saved ==========")
    print("summary:", summary_path)
    print("block_grad_client_summary:", result_dir / "block_grad_client_summary.csv")
    print("block_M_client_summary:", result_dir / "block_M_client_summary.csv")
    print("block_score_client_summary:", result_dir / "block_score_client_summary.csv")


if __name__ == "__main__":
    main()
