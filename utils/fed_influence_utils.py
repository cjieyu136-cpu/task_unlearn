"""
utils/fed_influence_utils.py

Fed-TA-MU influence utilities.

This module is responsible for gradient/IHVP/score computation only.
It does not load data, solve eps, apply repair, or evaluate final metrics.

Supported modes:
    bus_group   : Stage 3D bus-group shared-theta Fed-VJP + repo-style IHVP/score
    local_head  : Stage 3E client-local output-head Fed-VJP + repo-style IHVP/score
    block_Hk    : Stage 3F client-local Hessian H_k / block score
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from utils.reweight_utils import (
    compute_inverse_hvp_vector,
    compute_sample_scores,
    compute_test_gradient_repo_style,
)
from utils.fed_vjp_utils import (
    centralized_cost_gradient_repo_style,
    alignment_metrics,
)
from utils.fed_bus_client import compare_bus_fed_to_central_gradient
from utils.fed_local_head import local_head_cost_gradient_model_order
from utils.fed_block_influence import (
    block_influence_audit,
    canonical_vec_to_model_order,
)
from utils.fed_affine_analytic import (
    solve_affine_inverse_hvp_exact,
    compute_affine_sample_scores_exact,
)


def compute_global_repo_influence(
    cfg,
    ctx: Dict[str, Any],
    dataset_test_affine,
    dataset_remain,
    dataset_score,
    repair_criteria: str = "cost",
) -> Dict[str, Any]:
    """Compute repo-style global cost gradient, IHVP, and sample scores."""
    common = ctx["common"]
    model_ori = ctx["model_ori"]
    batch_size = common["batch_size"]
    train_loss = common["train_loss"]

    loader_score = DataLoader(dataset_score, batch_size=batch_size, shuffle=False)
    loader_remain = DataLoader(dataset_remain, batch_size=batch_size, shuffle=False)

    repair_criteria = str(repair_criteria).lower()
    if repair_criteria == "cost":
        grad_global = centralized_cost_gradient_repo_style(
            cfg=cfg,
            model=model_ori,
            dataset=dataset_test_affine,
            batch_size=batch_size,
        )
    else:
        grad_global = compute_test_gradient_repo_style(
            cfg=cfg,
            model_test=model_ori,
            # Match eval_unchange.py:
            # module_test uses loader_train = full train dataset, not remain.
            loader_train=loader_score,
            loader_test=DataLoader(dataset_test_affine, batch_size=batch_size, shuffle=False),
            dataset_test=dataset_test_affine,
            repair_criteria=repair_criteria,
            train_loss=train_loss,
        ).detach().cpu().numpy().reshape(-1).astype(float)
    exact_affine_hessian = bool(
        str(ctx.get("extra_info", {}).get("feature_mode", "precomputed_local_cache")) == "topology_local_fusion"
        and bool(getattr(cfg, "force_exact_affine_ihvp", False))
    )

    if exact_affine_hessian:
        analytic = solve_affine_inverse_hvp_exact(
            model=model_ori,
            dataset_remain=dataset_remain,
            grad_model_order=grad_global,
            damping=float(common.get("block_damping", 1e-8)),
        )
        M_global = analytic["M_model_order"]
        hvp_info_global = analytic["info"]

        score_exact = compute_affine_sample_scores_exact(
            model=model_ori,
            dataset_score=dataset_score,
            M_model_order=M_global,
            normalize_by_n=True,
        )
        scores_global = score_exact["scores"]
        score_info_global = score_exact["info"]
    else:
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
            loader_score=loader_score,
            dataset_score=dataset_score,
            M_vec=M_global,
            train_loss=train_loss,
            normalize_by_n=True,
        )

    return {
        "grad_global": grad_global,
        "M_global": M_global,
        "scores_global": scores_global,
        "hvp_info_global": hvp_info_global,
        "score_info_global": score_info_global,
    }


def compute_fed_influence(
    cfg,
    ctx: Dict[str, Any],
    dataset_test_affine,
    dataset_remain,
    dataset_score,
    repair_criteria: str = "cost",
) -> Dict[str, Any]:
    """
    Compute global baseline influence and Fed influence for the selected mode.

    Returns arrays:
        grad_global, grad_fed, M_global, M_fed, scores_global, scores_fed

    and alignments:
        grad_alignment, M_alignment, score_alignment
    """
    common = ctx["common"]
    fed_mode = common["fed_mode"]
    model_ori = ctx["model_ori"]
    batch_size = common["batch_size"]
    train_loss = common["train_loss"]

    global_inf = compute_global_repo_influence(
        cfg=cfg,
        ctx=ctx,
        dataset_test_affine=dataset_test_affine,
        dataset_remain=dataset_remain,
        dataset_score=dataset_score,
        repair_criteria=repair_criteria,
    )

    grad_global = global_inf["grad_global"]
    M_global = global_inf["M_global"]
    scores_global = global_inf["scores_global"]

    loader_score = DataLoader(dataset_score, batch_size=batch_size, shuffle=False)
    loader_remain = DataLoader(dataset_remain, batch_size=batch_size, shuffle=False)

    client_tables = {}
    extra_info = {}
    bus_groups_string = ctx["bus_groups_string"]

    if fed_mode == "bus_group":
        fed = compare_bus_fed_to_central_gradient(
            cfg=cfg,
            model=model_ori,
            dataset=dataset_test_affine,
            num_clients=common["num_bus_clients"],
            bus_groups=common["bus_groups_override"],
            batch_size=batch_size,
        )

        grad_fed = fed["grad_fed_bus"]

        M_fed, hvp_info_fed = compute_inverse_hvp_vector(
            cfg=cfg,
            model_train=model_ori,
            loader_train=loader_remain,
            vec=grad_fed,
            train_loss=train_loss,
            sign=-1.0,
        )

        scores_fed, score_info_fed = compute_sample_scores(
            cfg=cfg,
            model_train=model_ori,
            loader_hessian=loader_remain,
            loader_score=loader_score,
            dataset_score=dataset_score,
            M_vec=M_fed,
            train_loss=train_loss,
            normalize_by_n=True,
        )

        client_tables["bus_client_payload"] = pd.DataFrame(fed["per_client"])
        bus_groups_string = fed["bus_groups_string"]
        extra_info["fed_bus_info"] = fed.get("fed_info", {})

    elif fed_mode == "local_head":
        fed = local_head_cost_gradient_model_order(
            cfg=cfg,
            global_model=model_ori,
            dataset=dataset_test_affine,
            num_clients=common["num_bus_clients"],
            bus_groups=common["bus_groups_override"],
            batch_size=batch_size,
        )

        grad_fed = fed["local_grad_model_order"]

        M_fed, hvp_info_fed = compute_inverse_hvp_vector(
            cfg=cfg,
            model_train=model_ori,
            loader_train=loader_remain,
            vec=grad_fed,
            train_loss=train_loss,
            sign=-1.0,
        )

        scores_fed, score_info_fed = compute_sample_scores(
            cfg=cfg,
            model_train=model_ori,
            loader_hessian=loader_remain,
            loader_score=loader_score,
            dataset_score=dataset_score,
            M_vec=M_fed,
            train_loss=train_loss,
            normalize_by_n=True,
        )

        client_tables["local_head_client_payload"] = pd.DataFrame(fed["per_client"])
        bus_groups_string = fed["bus_groups_string"]
        extra_info.update({
            "prediction_alignment": fed["prediction_alignment"],
            "canonical_vjp_alignment": fed["vjp_alignment"],
            "model_order_alignment": fed["model_order_alignment"],
            "local_head_info": fed["info"],
        })

    elif fed_mode == "block_Hk":
        block = block_influence_audit(
            cfg=cfg,
            model=model_ori,
            dataset_test=dataset_test_affine,
            dataset_remain=dataset_remain,
            dataset_score=dataset_score,
            M_global_model_order=M_global,
            scores_global=scores_global,
            num_clients=common["num_bus_clients"],
            bus_groups=common["bus_groups_override"],
            batch_size=batch_size,
            block_damping=common["block_damping"],
            repair_criteria=repair_criteria,
        )

        feature_dim = int(dataset_score.feature.shape[1])
        output_dim = int(dataset_score.target.shape[1])
        grad_fed = canonical_vec_to_model_order(
            model=model_ori,
            feature_dim=feature_dim,
            output_dim=output_dim,
            canonical_vec=block["grad_result"]["grad_canonical"],
        )

        M_fed = block["M_block_model_order"]
        scores_fed = block["scores_block"]["scores"]
        hvp_info_fed = block["M_block"]["info"]
        score_info_fed = block["scores_block"]["info"]

        client_tables["block_grad_client_summary"] = pd.DataFrame(block["grad_result"]["per_client"])
        client_tables["block_M_client_summary"] = pd.DataFrame(block["M_block"]["per_client"])
        client_tables["block_score_client_summary"] = pd.DataFrame(block["scores_block"]["per_client"])

        bus_groups_string = block["info"]["bus_groups"]
        extra_info.update({
            "block_info": block["info"],
            "M_block_info": block["M_block"]["info"],
        })

    else:
        raise ValueError(
            f"Unsupported fed_mode={fed_mode}. Supported: bus_group, local_head, block_Hk"
        )

    if fed_mode in ["bus_group", "local_head"] and str(repair_criteria).lower() != "cost":
        raise NotImplementedError(
            f"repair_criteria={repair_criteria} is currently implemented for fed_mode=block_Hk only."
        )

    grad_alignment = alignment_metrics(grad_global, grad_fed)
    M_alignment = alignment_metrics(M_global, M_fed)
    score_alignment = alignment_metrics(scores_global, scores_fed)

    return {
        "fed_mode": fed_mode,
        "bus_groups": bus_groups_string,
        "grad_global": grad_global,
        "grad_fed": grad_fed,
        "M_global": M_global,
        "M_fed": M_fed,
        "scores_global": scores_global,
        "scores_fed": scores_fed,
        "hvp_info_global": global_inf["hvp_info_global"],
        "hvp_info_fed": hvp_info_fed,
        "score_info_global": global_inf["score_info_global"],
        "score_info_fed": score_info_fed,
        "grad_alignment": grad_alignment,
        "M_alignment": M_alignment,
        "score_alignment": score_alignment,
        "client_tables": client_tables,
        "extra_info": extra_info,
    }


def save_influence_outputs(result_dir, influence: Dict[str, Any]) -> None:
    """Save common influence arrays and client summary tables."""
    result_dir.mkdir(parents=True, exist_ok=True)

    np.save(result_dir / "grad_global.npy", influence["grad_global"])
    np.save(result_dir / "grad_fed.npy", influence["grad_fed"])
    np.save(result_dir / "M_global.npy", influence["M_global"])
    np.save(result_dir / "M_fed.npy", influence["M_fed"])
    np.save(result_dir / "scores_global.npy", influence["scores_global"])
    np.save(result_dir / "scores_fed.npy", influence["scores_fed"])

    for name, df in influence["client_tables"].items():
        df.to_csv(result_dir / f"{name}.csv", index=False)
