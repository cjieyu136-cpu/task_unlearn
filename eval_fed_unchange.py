"""
eval_fed_unchange.py

Fed-TA-MU performance-unchanged repair entrypoint with Stage 3H H2-2 runtime
and H3-1 secure aggregation mock support.

Main mode:
python eval_fed_unchange.py model=conv unlearn_prop=0.2 +index_mode=helpful +index_criteria=cost +rho=0.001 +fed_mode=block_Hk +num_bus_clients=4 +feature_mode=fusion_topology_only +runtime_mode=precomputed_local_cache +secure_agg_mode=none

Recommended explicit topology controls:
    +topology_partition_enabled=true
    +topology_fusion_enabled=true
    +topology_repair_regularization_enabled=true
    +topology_encoder_propagation_enabled=false
    +server_feature_layout=secure_agg_mean

Supported feature_mode:
    precomputed_local_cache
    topology_local_fusion
    fusion_topology_only

Supported runtime_mode:
    precomputed_local_cache
    local_frozen_backbone

Supported secure_agg_mode:
    none
    mock_sum
"""

from pathlib import Path
import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.index_utils import (
    load_and_split_unlearn_datasets,
    load_saved_unlearn_object,
    save_unlearn_object,
)
from utils.fed_data_utils import (
    prepare_repo_affine_context,
    describe_context,
    build_result_dir,
    resolve_feature_and_runtime_mode,
    compact_feature_mode_for_path,
    compact_runtime_mode_for_path,
)
from utils.fed_influence_utils import (
    compute_fed_influence,
    save_influence_outputs,
    compute_global_repo_influence,
)
from utils.fed_repair_utils import (
    evaluate_all,
    build_summary_row,
    run_complete_unlearning_baseline,
    run_fed_repair_grid,
)
from utils.fed_server_runtime import (
    build_runtime_server_from_affine_context,
    build_runtime_server_from_raw_context,
    build_runtime_topology_servers_from_raw_context,
)
from utils.fed_vjp_utils import alignment_metrics
from utils.fed_local_head import canonical_vec_to_model_order
from utils.fed_secure_aggregation import MockSecureSumAggregator, redact_client_table
from utils.fed_vjp_utils import (
    cloud_criterion_gradient_scaled_for_dataset,
    cloud_safety_proxy_for_dataset,
)
from func_operation import return_nn_model


def _messages_to_df(messages):
    rows = []
    for msg in messages:
        row = {
            "client_id": msg.client_id,
            "message_type": msg.message_type,
            "payload_shape": msg.payload_shape,
            "num_elements": msg.num_elements,
        }
        row.update({f"meta_{k}": v for k, v in msg.metadata.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_runtime_test_gradient(server_test, split_name="test", secure_agg_mode="none"):
    output_dim = server_test.output_dim
    feature_dim = server_test.feature_dim

    W_grad_full = np.zeros((output_dim, feature_dim), dtype=np.float64)
    b_grad_full = np.zeros((output_dim,), dtype=np.float64)

    grad_rows = []
    messages = []
    secure = MockSecureSumAggregator(mode=secure_agg_mode)

    for client in server_test.clients:
        g = client.compute_criterion_gradient(split_name=split_name)
        messages.append(g["message"])

        buses = client.bus_indices
        W_grad_full[buses, :] = g["W_grad"]
        b_grad_full[buses] = g["b_grad"]

        # Secure-sum mock requires same vector shape from every client.
        # Local bus blocks have different sizes, so pad each client contribution
        # into the full canonical affine-head vector before adding.
        W_pad = np.zeros((output_dim, feature_dim), dtype=np.float64)
        b_pad = np.zeros((output_dim,), dtype=np.float64)
        W_pad[buses, :] = g["W_grad"]
        b_pad[buses] = g["b_grad"]
        grad_pad_vec = np.concatenate([W_pad.reshape(-1), b_pad.reshape(-1)])
        secure.add(
            "cost_gradient",
            client.client_id,
            grad_pad_vec,
            metadata={"kind": "zero-padded client cost-gradient canonical vector", "feature_mode": client.feature_mode},
        )

        meta = client.local_metadata()
        grad_rows.append({
            **meta,
            "g_k_norm": float(np.linalg.norm(g["grad_vec"])),
            "g_k_elements": int(g["grad_vec"].size),
        })

    grad_canonical = np.concatenate([W_grad_full.reshape(-1), b_grad_full.reshape(-1)])
    grad_df = pd.DataFrame(grad_rows)
    grad_df = redact_client_table(grad_df, secure_agg_mode)

    return grad_canonical, W_grad_full, b_grad_full, grad_df, messages, secure


def _compute_runtime_train_M_and_scores(
    server_train,
    W_grad_full,
    b_grad_full,
    remain_index,
    block_damping,
    secure_agg_mode="none",
    client_score_weights=None,
    client_bus_score_weights=None,
):
    output_dim = server_train.output_dim
    feature_dim = server_train.feature_dim

    W_M_full = np.zeros((output_dim, feature_dim), dtype=np.float64)
    b_M_full = np.zeros((output_dim,), dtype=np.float64)

    runtime_scores = np.zeros((len(server_train.dataset_reference),), dtype=np.float64)

    M_rows = []
    score_rows = []
    messages = []

    secure = MockSecureSumAggregator(mode=secure_agg_mode)
    remain_index = np.asarray(remain_index, dtype=int)

    for client in server_train.clients:
        buses = client.bus_indices

        M = client.compute_block_ihvp(
            W_grad=W_grad_full[buses, :],
            b_grad=b_grad_full[buses],
            indices=remain_index,
            damping=block_damping,
        )
        messages.append(M["message"])

        score = client.compute_score_contribution(
            W_M=M["W_M"],
            b_M=M["b_M"],
            indices=None,
            normalize_by_n=True,
            bus_weight=None if client_bus_score_weights is None else client_bus_score_weights[client.client_id],
        )
        messages.append(score["message"])
        score_local = np.asarray(score["score"], dtype=np.float64)
        weight_mean = np.nan
        weight_std = np.nan
        bus_weight_mean = np.nan
        bus_weight_std = np.nan
        if client_score_weights is not None:
            local_weight = np.asarray(client_score_weights[client.client_id], dtype=np.float64).reshape(-1)
            score_local = score_local * local_weight
            weight_mean = float(np.mean(local_weight))
            weight_std = float(np.std(local_weight))
        if client_bus_score_weights is not None:
            local_bus_weight = np.asarray(client_bus_score_weights[client.client_id], dtype=np.float64)
            bus_weight_mean = float(np.mean(local_bus_weight))
            bus_weight_std = float(np.std(local_bus_weight))

        W_M_full[buses, :] = M["W_M"]
        b_M_full[buses] = M["b_M"]
        runtime_scores += score_local

        # Secure-sum mock requires same vector shape from every client.
        # Local M_k blocks have different sizes, so pad each client contribution
        # into the full canonical affine-head vector before adding.
        W_M_pad = np.zeros((output_dim, feature_dim), dtype=np.float64)
        b_M_pad = np.zeros((output_dim,), dtype=np.float64)
        W_M_pad[buses, :] = M["W_M"]
        b_M_pad[buses] = M["b_M"]
        M_pad_vec = np.concatenate([W_M_pad.reshape(-1), b_M_pad.reshape(-1)])
        secure.add(
            "block_ihvp",
            client.client_id,
            M_pad_vec,
            metadata={"kind": "zero-padded client M_k canonical vector", "feature_mode": client.feature_mode},
        )
        secure.add(
            "score_contribution",
            client.client_id,
            score_local,
            metadata={
                "kind": "client sample-score contribution",
                "feature_mode": client.feature_mode,
                "shadow_weight_mean": weight_mean,
                "shadow_bus_weight_mean": bus_weight_mean,
            },
        )

        meta = client.local_metadata()
        M_rows.append({
            **meta,
            "M_k_norm": float(np.linalg.norm(M["M_vec"])),
            "M_k_elements": int(M["M_vec"].size),
            "H_condition": float(M["H_condition"]),
            "solver": M["solver"],
        })
        score_rows.append({
            **meta,
            "score_min": float(np.min(score_local)),
            "score_max": float(np.max(score_local)),
            "score_mean": float(np.mean(score_local)),
            "score_std": float(np.std(score_local)),
            "shadow_weight_mean": weight_mean,
            "shadow_weight_std": weight_std,
            "shadow_bus_weight_mean": bus_weight_mean,
            "shadow_bus_weight_std": bus_weight_std,
        })

    M_canonical = np.concatenate([W_M_full.reshape(-1), b_M_full.reshape(-1)])

    M_df = redact_client_table(pd.DataFrame(M_rows), secure_agg_mode)
    score_df = redact_client_table(pd.DataFrame(score_rows), secure_agg_mode)

    return M_canonical, runtime_scores, M_df, score_df, messages, secure


def compute_runtime_block_influence_for_eval(
    cfg,
    ctx,
    dataset_test_affine,
    dataset_remain,
    dataset_score,
    unlearn_obj,
    result_dir,
    repair_criteria="cost",
    client_score_weights=None,
    client_bus_score_weights=None,
):
    common = ctx["common"]
    mode_info = resolve_feature_and_runtime_mode(cfg)
    runtime_input_mode = mode_info["feature_mode"]
    runtime_mode = mode_info["runtime_mode"]
    secure_agg_mode = str(OmegaConf.select(cfg, "secure_agg_mode", default="none"))
    device = str(OmegaConf.select(cfg, "device", default="cpu"))
    feature_batch_size = int(OmegaConf.select(cfg, "fed_feature_batch_size", default=128))

    if common["fed_mode"] != "block_Hk":
        raise ValueError("runtime feature mode currently supports fed_mode=block_Hk only.")
    if secure_agg_mode not in ["none", "mock_sum"]:
        raise ValueError("secure_agg_mode must be none or mock_sum.")

    runtime_messages = []
    runtime_extra_tables = {}
    runtime_topology_info = {}
    topology_info = ctx.get("extra_info", {})
    topology_fusion_enabled = bool(topology_info.get("topology_fusion_enabled", False))

    if runtime_mode == "precomputed_local_cache":
        server_train = build_runtime_server_from_affine_context(
            cfg=cfg,
            ctx=ctx,
            dataset_for_clients=dataset_score,
            feature_mode=str(ctx.get("extra_info", {}).get("feature_mode", runtime_input_mode)),
        )
        server_test = build_runtime_server_from_affine_context(
            cfg=cfg,
            ctx=ctx,
            dataset_for_clients=dataset_test_affine,
            feature_mode=str(ctx.get("extra_info", {}).get("feature_mode", runtime_input_mode)),
        )
    elif runtime_mode == "local_frozen_backbone":
        frozen_backbone = return_nn_model(cfg, is_load=True, dataset="core")
        frozen_backbone.eval()
        if topology_fusion_enabled:
            runtime_topology = build_runtime_topology_servers_from_raw_context(
                cfg=cfg,
                ctx=ctx,
                raw_train_dataset=ctx["dataset_sensitive"],
                raw_test_dataset=ctx["dataset_test"],
                frozen_backbone=frozen_backbone,
                device=device,
            )
            server_train = runtime_topology["server_train"]
            server_test = runtime_topology["server_test"]
            dataset_score = runtime_topology["dataset_train_affine"]
            dataset_test_affine = runtime_topology["dataset_test_affine"]
            runtime_extra_tables.update(runtime_topology["extra_tables"])
            runtime_topology_info.update(runtime_topology["runtime_topology_info"])
        else:
            server_train = build_runtime_server_from_raw_context(
                cfg=cfg,
                ctx=ctx,
                raw_dataset_for_clients=ctx["dataset_sensitive"],
                target_dataset_reference=dataset_score,
                frozen_backbone=frozen_backbone,
                feature_batch_size=feature_batch_size,
                device=device,
            )
            server_test = build_runtime_server_from_raw_context(
                cfg=cfg,
                ctx=ctx,
                raw_dataset_for_clients=ctx["dataset_test"],
                target_dataset_reference=dataset_test_affine,
                frozen_backbone=frozen_backbone,
                feature_batch_size=feature_batch_size,
                device=device,
            )
    else:
        raise ValueError("runtime_mode must be precomputed_local_cache or local_frozen_backbone.")

    global_inf = compute_global_repo_influence(
        cfg=cfg,
        ctx=ctx,
        dataset_test_affine=dataset_test_affine,
        dataset_remain=dataset_remain,
        dataset_score=dataset_score,
        repair_criteria=repair_criteria,
    )

    train_feature_summary = server_train.prepare_client_features()
    test_feature_summary = server_test.prepare_client_features()
    train_feature_summary.to_csv(result_dir / "runtime_train_feature_summary.csv", index=False)
    test_feature_summary.to_csv(result_dir / "runtime_test_feature_summary.csv", index=False)

    server_test.compute_cloud_gy_and_dispatch(
        repair_criteria=repair_criteria,
        split_name="test",
        batch_size=common["batch_size"],
        device=device,
    )

    grad_canonical, W_grad_full, b_grad_full, grad_df, grad_messages, grad_secure = _compute_runtime_test_gradient(
        server_test,
        split_name="test",
        secure_agg_mode=secure_agg_mode,
    )
    runtime_messages.extend(grad_messages)

    M_canonical, scores_fed, M_df, score_df, M_score_messages, train_secure = _compute_runtime_train_M_and_scores(
        server_train=server_train,
        W_grad_full=W_grad_full,
        b_grad_full=b_grad_full,
        remain_index=unlearn_obj["remain_index"],
        block_damping=common["block_damping"],
        secure_agg_mode=secure_agg_mode,
        client_score_weights=client_score_weights,
        client_bus_score_weights=client_bus_score_weights,
    )
    runtime_messages.extend(M_score_messages)

    feature_dim = int(dataset_score.feature.shape[1])
    output_dim = int(dataset_score.target.shape[1])
    model_ori = ctx["model_ori"]

    grad_fed = canonical_vec_to_model_order(
        model=model_ori,
        feature_dim=feature_dim,
        output_dim=output_dim,
        canonical_vec=grad_canonical,
    )
    M_fed = canonical_vec_to_model_order(
        model=model_ori,
        feature_dim=feature_dim,
        output_dim=output_dim,
        canonical_vec=M_canonical,
    )

    grad_alignment = alignment_metrics(global_inf["grad_global"], grad_fed)
    M_alignment = alignment_metrics(global_inf["M_global"], M_fed)
    score_alignment = alignment_metrics(global_inf["scores_global"], scores_fed)

    if secure_agg_mode == "mock_sum":
        # Public payload only includes aggregate records plus non-sensitive feature-ready / gy dispatch metadata.
        payload_df = pd.concat(
            [
                server_train.payload_summary()[lambda d: d["message_type"].isin(["local_feature_ready"])],
                server_test.payload_summary()[lambda d: d["message_type"].isin(["local_feature_ready", "receive_gy_slice"])],
                grad_secure.public_records(),
                train_secure.public_records(),
            ],
            ignore_index=True,
        )
        secure_summary_df = pd.concat(
            [grad_secure.public_records(), train_secure.public_records()],
            ignore_index=True,
        )
        secure_summary_df.to_csv(result_dir / "runtime_secure_agg_summary.csv", index=False)
    else:
        payload_df = pd.concat(
            [
                server_train.payload_summary(),
                server_test.payload_summary(),
                _messages_to_df(runtime_messages),
            ],
            ignore_index=True,
        )

    client_tables = {
        "runtime_grad_client_summary": grad_df,
        "runtime_M_client_summary": M_df,
        "runtime_score_client_summary": score_df,
        "runtime_payload_summary": payload_df,
    }
    for name, df in runtime_extra_tables.items():
        client_tables[f"runtime_{name}"] = df.copy()
    for name, df in ctx.get("extra_tables", {}).items():
        client_tables[f"runtime_{name}"] = df.copy()

    M_info = {
        "feature_mode": str(ctx.get("extra_info", {}).get("feature_mode", runtime_input_mode)),
        "runtime_input_mode": runtime_input_mode,
        "runtime_mode": runtime_mode,
        "secure_agg_mode": secure_agg_mode,
        "H_condition": np.nan if "H_condition" not in M_df.columns or len(M_df) == 0 else float(M_df["H_condition"].iloc[0]),
        "solver": "" if "solver" not in M_df.columns or len(M_df) == 0 else str(M_df["solver"].iloc[0]),
        "payload_total_elements": int(payload_df["num_elements"].sum()) if "num_elements" in payload_df.columns else np.nan,
    }

    if secure_agg_mode == "mock_sum":
        M_info.update(grad_secure.summary())
        M_info.update({f"train_{k}": v for k, v in train_secure.summary().items()})

    return {
        "fed_mode": common["fed_mode"],
        "feature_mode": str(ctx.get("extra_info", {}).get("feature_mode", runtime_input_mode)),
        "secure_agg_mode": secure_agg_mode,
        "bus_groups": ctx["bus_groups_string"],
        "grad_global": global_inf["grad_global"],
        "grad_fed": grad_fed,
        "M_global": global_inf["M_global"],
        "M_fed": M_fed,
        "scores_global": global_inf["scores_global"],
        "scores_fed": scores_fed,
        "hvp_info_global": global_inf["hvp_info_global"],
        "hvp_info_fed": M_info,
        "score_info_global": global_inf["score_info_global"],
        "score_info_fed": {
            "feature_mode": str(ctx.get("extra_info", {}).get("feature_mode", runtime_input_mode)),
            "runtime_input_mode": runtime_input_mode,
            "secure_agg_mode": secure_agg_mode,
        },
        "grad_alignment": grad_alignment,
        "M_alignment": M_alignment,
        "score_alignment": score_alignment,
        "client_tables": client_tables,
        "extra_info": {
            "runtime_info": M_info,
            "feature_mode": str(ctx.get("extra_info", {}).get("feature_mode", runtime_input_mode)),
            "runtime_input_mode": runtime_input_mode,
            "runtime_mode": runtime_mode,
            "secure_agg_mode": secure_agg_mode,
            "runtime_topology_info": runtime_topology_info,
            "topology_extra_info": ctx.get("extra_info", {}),
        },
    }


def save_runtime_or_standard_influence_outputs(result_dir, influence):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "grad_global.npy", influence["grad_global"])
    np.save(result_dir / "grad_fed.npy", influence["grad_fed"])
    np.save(result_dir / "M_global.npy", influence["M_global"])
    np.save(result_dir / "M_fed.npy", influence["M_fed"])
    np.save(result_dir / "scores_global.npy", influence["scores_global"])
    np.save(result_dir / "scores_fed.npy", influence["scores_fed"])

    for name, df in influence["client_tables"].items():
        df.to_csv(result_dir / f"{name}.csv", index=False)


def compute_shadow_price_sample_weights(
    cfg,
    model,
    dataset_score,
    result_dir,
    bus_groups=None,
    repair_criteria: str = "cost",
):
    mode = str(OmegaConf.select(cfg, "shadow_weight_mode", default="none"))
    repair_criteria = str(repair_criteria).lower()
    if mode in ["", "none", "off", "false"]:
        return {
            "sample_weights": None,
            "client_weight_matrix": None,
        }, {
            "shadow_weight_mode": "none",
            "shadow_weight_repair_criteria": repair_criteria,
        }

    w_min = float(OmegaConf.select(cfg, "shadow_weight_min", default=0.5))
    w_max = float(OmegaConf.select(cfg, "shadow_weight_max", default=2.0))
    tau = float(OmegaConf.select(cfg, "shadow_weight_tau", default=1.0))
    batch_size = int(OmegaConf.select(cfg, "batch_size", default=128))
    device = str(OmegaConf.select(cfg, "device", default="cpu"))

    cloud = cloud_criterion_gradient_scaled_for_dataset(
        cfg=cfg,
        model=model,
        dataset=dataset_score,
        repair_criteria=repair_criteria,
        batch_size=batch_size,
        device=device,
    )

    client_weight_matrix = None
    client_bus_weight_list = None

    hybrid_client_mode = mode in ["hybrid_client_mean_abs_gy", "hybrid_client_l2_gy"]
    softmax_client_mode = mode in ["softmax_client_mean_abs_gy", "softmax_client_l2_gy"]
    softmax_relative_client_mode = mode in [
        "softmax_relative_client_mean_abs_gy",
        "softmax_relative_client_l2_gy",
    ]
    bus_relative_mode = mode in ["bus_relative_abs_gy"]
    client_mode = mode in [
        "client_mean_abs_gy",
        "client_l2_gy",
        "hybrid_client_mean_abs_gy",
        "hybrid_client_l2_gy",
        "softmax_client_mean_abs_gy",
        "softmax_client_l2_gy",
        "softmax_relative_client_mean_abs_gy",
        "softmax_relative_client_l2_gy",
    ]

    if mode == "mean_abs_gy":
        raw = np.mean(np.abs(cloud["gy_scaled"]), axis=1)
    elif mode == "l2_gy":
        raw = np.sqrt(np.mean(np.square(cloud["gy_scaled"]), axis=1))
    elif mode == "first_order_proxy_abs":
        raw = np.abs(np.asarray(cloud["first_order_cost_proxy"], dtype=float))
    elif mode in [
        "client_mean_abs_gy",
        "hybrid_client_mean_abs_gy",
        "softmax_client_mean_abs_gy",
        "softmax_relative_client_mean_abs_gy",
    ]:
        raw = np.mean(np.abs(cloud["gy_scaled"]), axis=1)
    elif mode in [
        "client_l2_gy",
        "hybrid_client_l2_gy",
        "softmax_client_l2_gy",
        "softmax_relative_client_l2_gy",
    ]:
        raw = np.sqrt(np.mean(np.square(cloud["gy_scaled"]), axis=1))
    elif mode == "bus_relative_abs_gy":
        raw = np.mean(np.abs(cloud["gy_scaled"]), axis=1)
    else:
        raise ValueError(f"Unsupported shadow_weight_mode: {mode}")

    raw = np.asarray(raw, dtype=float).reshape(-1)
    raw_mean = float(np.mean(raw)) if raw.size else 1.0
    if not np.isfinite(raw_mean) or raw_mean <= 0:
        raw_mean = 1.0
    weights = raw / raw_mean
    weights = np.clip(weights, w_min, w_max).astype(np.float64)

    client_raw_matrix = None
    softmax_input_matrix = None
    if client_mode:
        if bus_groups is None:
            raise ValueError("bus_groups are required for client-level shadow weighting modes")
        client_rows = []
        gy_scaled = np.asarray(cloud["gy_scaled"], dtype=float)
        for buses in bus_groups:
            buses = np.asarray(buses, dtype=int)
            if mode in [
                "client_mean_abs_gy",
                "hybrid_client_mean_abs_gy",
                "softmax_client_mean_abs_gy",
                "softmax_relative_client_mean_abs_gy",
            ]:
                local_raw = np.mean(np.abs(gy_scaled[:, buses]), axis=1)
            else:
                local_raw = np.sqrt(np.mean(np.square(gy_scaled[:, buses]), axis=1))
            if hybrid_client_mode or softmax_client_mode or softmax_relative_client_mode:
                client_rows.append(np.asarray(local_raw, dtype=np.float64))
            else:
                local_mean = float(np.mean(local_raw)) if local_raw.size else 1.0
                if not np.isfinite(local_mean) or local_mean <= 0:
                    local_mean = 1.0
                local_weight = np.clip(local_raw / local_mean, w_min, w_max).astype(np.float64)
                client_rows.append(local_weight)
        if hybrid_client_mode:
            client_raw_matrix = np.stack(client_rows, axis=0)
            per_sample_mean = np.mean(client_raw_matrix, axis=0, keepdims=True)
            per_sample_mean[~np.isfinite(per_sample_mean)] = 1.0
            per_sample_mean[per_sample_mean <= 0] = 1.0
            client_weight_matrix = np.clip(client_raw_matrix / per_sample_mean, w_min, w_max).astype(np.float64)
        elif softmax_client_mode or softmax_relative_client_mode:
            client_raw_matrix = np.stack(client_rows, axis=0)
            softmax_input_matrix = client_raw_matrix
            if softmax_relative_client_mode:
                per_sample_mean = np.mean(client_raw_matrix, axis=0, keepdims=True)
                per_sample_mean[~np.isfinite(per_sample_mean)] = 1.0
                per_sample_mean[per_sample_mean <= 0] = 1.0
                softmax_input_matrix = client_raw_matrix / per_sample_mean
            tau_safe = tau if np.isfinite(tau) and tau > 0 else 1.0
            logits = softmax_input_matrix / tau_safe
            logits = logits - np.max(logits, axis=0, keepdims=True)
            exp_logits = np.exp(logits)
            denom = np.sum(exp_logits, axis=0, keepdims=True)
            denom[~np.isfinite(denom)] = 1.0
            denom[denom <= 0] = 1.0
            client_weight_matrix = exp_logits / denom
        else:
            client_raw_matrix = np.stack(client_rows, axis=0)
            client_weight_matrix = client_raw_matrix
    elif bus_relative_mode:
        if bus_groups is None:
            raise ValueError("bus_groups are required for bus-level shadow weighting modes")
        gy_scaled = np.asarray(cloud["gy_scaled"], dtype=float)
        client_bus_weight_list = []
        for buses in bus_groups:
            buses = np.asarray(buses, dtype=int)
            local_abs = np.abs(gy_scaled[:, buses])
            per_sample_mean = np.mean(local_abs, axis=1, keepdims=True)
            per_sample_mean[~np.isfinite(per_sample_mean)] = 1.0
            per_sample_mean[per_sample_mean <= 0] = 1.0
            local_bus_weight = np.clip(local_abs / per_sample_mean, w_min, w_max).astype(np.float64)
            client_bus_weight_list.append(local_bus_weight)

    summary = {
        "shadow_weight_mode": mode,
        "shadow_weight_repair_criteria": repair_criteria,
        "shadow_weight_min": w_min,
        "shadow_weight_max": w_max,
        "shadow_weight_tau": tau,
        "shadow_weight_raw_mean": raw_mean,
        "shadow_weight_final_mean": float(np.mean(weights)),
        "shadow_weight_final_std": float(np.std(weights)),
        "shadow_weight_final_min": float(np.min(weights)),
        "shadow_weight_final_max": float(np.max(weights)),
        "shadow_weight_num_samples": int(weights.size),
    }
    if client_weight_matrix is not None:
        summary.update({
            "shadow_client_weight_mean": float(np.mean(client_weight_matrix)),
            "shadow_client_weight_std": float(np.std(client_weight_matrix)),
            "shadow_client_weight_min": float(np.min(client_weight_matrix)),
            "shadow_client_weight_max": float(np.max(client_weight_matrix)),
            "shadow_num_clients": int(client_weight_matrix.shape[0]),
            "shadow_client_raw_std": float(np.std(client_raw_matrix)) if client_raw_matrix is not None else 0.0,
            "shadow_client_weight_scheme": (
                "per_sample_relative"
                if hybrid_client_mode
                else (
                    "softmax_relative_across_client"
                    if softmax_relative_client_mode
                    else ("softmax_across_client" if softmax_client_mode else "per_client_normalized")
                )
            ),
        })
        if softmax_client_mode or softmax_relative_client_mode:
            summary["shadow_softmax_input_std"] = float(np.std(softmax_input_matrix))
    if client_bus_weight_list is not None:
        all_bus_weight = np.concatenate([w.reshape(-1) for w in client_bus_weight_list], axis=0)
        summary.update({
            "shadow_bus_weight_mean": float(np.mean(all_bus_weight)),
            "shadow_bus_weight_std": float(np.std(all_bus_weight)),
            "shadow_bus_weight_min": float(np.min(all_bus_weight)),
            "shadow_bus_weight_max": float(np.max(all_bus_weight)),
            "shadow_bus_weight_scheme": "per_sample_relative_within_client_bus",
        })

    np.save(result_dir / "shadow_sample_weights.npy", weights)
    pd.DataFrame({
        "sample_index": np.arange(weights.size, dtype=int),
        "shadow_weight": weights,
        "shadow_weight_raw": raw,
    }).to_csv(result_dir / "shadow_sample_weights.csv", index=False)
    pd.DataFrame([summary]).to_csv(result_dir / "shadow_weight_summary.csv", index=False)
    if client_weight_matrix is not None:
        np.save(result_dir / "shadow_client_weights.npy", client_weight_matrix)
        client_weight_df = pd.DataFrame({"sample_index": np.arange(client_weight_matrix.shape[1], dtype=int)})
        for client_id in range(client_weight_matrix.shape[0]):
            client_weight_df[f"client_{client_id}"] = client_weight_matrix[client_id]
        client_weight_df.to_csv(result_dir / "shadow_client_weights.csv", index=False)
    if client_bus_weight_list is not None:
        for client_id, local_bus_weight in enumerate(client_bus_weight_list):
            np.save(result_dir / f"shadow_bus_weights_client_{client_id}.npy", local_bus_weight)

    return {
        "sample_weights": weights,
        "client_weight_matrix": client_weight_matrix,
        "client_bus_weight_list": client_bus_weight_list,
    }, summary


def compute_safety_sample_gate(cfg, model, dataset_score, result_dir):
    mode = str(OmegaConf.select(cfg, "safety_weight_mode", default="none"))
    if mode in ["", "none", "off", "false"]:
        return {"sample_risk": None, "sample_gate": None}, {"safety_weight_mode": "none"}

    if mode != "stage2_ls":
        raise ValueError(f"Unsupported safety_weight_mode: {mode}")

    beta = float(OmegaConf.select(cfg, "safety_weight_beta", default=0.5))
    g_min = float(OmegaConf.select(cfg, "safety_weight_min", default=0.5))
    g_max = float(OmegaConf.select(cfg, "safety_weight_max", default=1.0))
    batch_size = int(OmegaConf.select(cfg, "batch_size", default=128))
    device = str(OmegaConf.select(cfg, "device", default="cpu"))

    safety = cloud_safety_proxy_for_dataset(
        cfg=cfg,
        model=model,
        dataset=dataset_score,
        batch_size=batch_size,
        device=device,
    )

    raw_risk = np.asarray(safety["stage2_ls_total"], dtype=float).reshape(-1)
    risk_mean = float(np.mean(raw_risk)) if raw_risk.size else 1.0
    if not np.isfinite(risk_mean) or risk_mean <= 0:
        risk_mean = 1.0
    risk_norm = raw_risk / risk_mean
    gate = 1.0 / (1.0 + beta * risk_norm)
    gate = np.clip(gate, g_min, g_max).astype(np.float64)

    summary = {
        "safety_weight_mode": mode,
        "safety_weight_beta": beta,
        "safety_weight_min": g_min,
        "safety_weight_max": g_max,
        "safety_risk_mean": risk_mean,
        "safety_gate_mean": float(np.mean(gate)),
        "safety_gate_std": float(np.std(gate)),
        "safety_gate_min": float(np.min(gate)),
        "safety_gate_max": float(np.max(gate)),
        "safety_weight_num_samples": int(gate.size),
        "safety_stage2_ls_mean": safety["info"]["stage2_ls_mean"],
        "safety_stage2_ls_max": safety["info"]["stage2_ls_max"],
        "safety_stage2_gs_mean": safety["info"]["stage2_gs_mean"],
        "safety_dispatch_gap_abs_mean": safety["info"]["dispatch_gap_abs_mean"],
    }

    np.save(result_dir / "safety_sample_risk.npy", raw_risk)
    pd.DataFrame({
        "sample_index": np.arange(raw_risk.size, dtype=int),
        "safety_risk": raw_risk,
        "safety_risk_norm": risk_norm,
        "safety_gate": gate,
    }).to_csv(result_dir / "safety_sample_risk.csv", index=False)
    pd.DataFrame([summary]).to_csv(result_dir / "safety_weight_summary.csv", index=False)

    return {
        "sample_risk": raw_risk,
        "sample_gate": gate,
    }, summary


def compute_safety_sample_risk(cfg, model, dataset_score, result_dir, risk_mode="stage2_ls"):
    if str(risk_mode).lower() != "stage2_ls":
        raise ValueError(f"Unsupported joint_risk_mode: {risk_mode}")

    batch_size = int(OmegaConf.select(cfg, "batch_size", default=128))
    device = str(OmegaConf.select(cfg, "device", default="cpu"))
    safety = cloud_safety_proxy_for_dataset(
        cfg=cfg,
        model=model,
        dataset=dataset_score,
        batch_size=batch_size,
        device=device,
    )
    raw_risk = np.asarray(safety["stage2_ls_total"], dtype=float).reshape(-1)
    pd.DataFrame({
        "sample_index": np.arange(raw_risk.size, dtype=int),
        "safety_risk": raw_risk,
    }).to_csv(result_dir / "joint_safety_sample_risk.csv", index=False)
    np.save(result_dir / "joint_safety_sample_risk.npy", raw_risk)
    return raw_risk


def compute_joint_risk_score(raw_risk, score_reference):
    raw_risk = np.asarray(raw_risk, dtype=np.float64).reshape(-1)
    score_reference = np.asarray(score_reference, dtype=np.float64).reshape(-1)

    risk_mean = float(np.mean(raw_risk)) if raw_risk.size else 1.0
    if not np.isfinite(risk_mean) or risk_mean <= 0:
        risk_mean = 1.0
    risk_norm = raw_risk / risk_mean

    score_scale = float(np.mean(np.abs(score_reference))) if score_reference.size else 1.0
    if not np.isfinite(score_scale) or score_scale <= 0:
        score_scale = 1.0

    risk_score = (risk_norm * score_scale).astype(np.float64)
    return {
        "raw_risk": raw_risk,
        "risk_norm": risk_norm.astype(np.float64),
        "risk_score": risk_score,
        "risk_mean": risk_mean,
        "score_scale": score_scale,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("========== eval_fed_unchange: Fed-TA-MU repair ==========")

    ctx = prepare_repo_affine_context(cfg)
    common = ctx["common"]
    repair_criteria = str(OmegaConf.select(cfg, "repair_criteria", default=cfg.criteria)).lower()
    reuse_unlearn_dir = OmegaConf.select(cfg, "reuse_unlearn_dir", default=None)

    mode_info = resolve_feature_and_runtime_mode(cfg)
    requested_feature_mode = mode_info["requested_feature_mode"]
    runtime_input_mode = mode_info["feature_mode"]
    affine_feature_mode = str(ctx.get("extra_info", {}).get("feature_mode", runtime_input_mode))
    runtime_mode = mode_info["runtime_mode"]
    secure_agg_mode = str(OmegaConf.select(cfg, "secure_agg_mode", default="none"))
    use_runtime = bool(OmegaConf.select(cfg, "fed_runtime", default=True))

    print("---------- context ----------")
    for k, v in describe_context(ctx).items():
        print(f"{k}: {v}")
    print("affine_feature_mode:", affine_feature_mode)
    print("runtime_input_mode:", runtime_input_mode)
    print("requested_feature_mode:", requested_feature_mode)
    print("runtime_mode:", runtime_mode)
    print("fed_runtime:", use_runtime)
    print("secure_agg_mode:", secure_agg_mode)

    result_dir = build_result_dir(
        cfg=cfg,
        model_type=common["model_type"],
        fed_mode=common["fed_mode"],
        eval_kind="unlearn",
        index_mode=common["index_mode"],
        index_criteria=common["index_criteria"],
    ).resolve()

    result_dir = (
        result_dir.parent.parent
        / "unchange_repair_runtime"
        / compact_feature_mode_for_path(affine_feature_mode)
        / f"rt_{compact_runtime_mode_for_path(runtime_mode)}"
        / f"secure_{secure_agg_mode}"
        / result_dir.name
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    print("result_dir:", result_dir)

    dataset_train_affine = ctx["dataset_train_affine"]
    dataset_test_affine = ctx["dataset_test_affine"]
    model_ori = ctx["model_ori"]
    parameter_ori = ctx["parameter_ori"]

    if reuse_unlearn_dir:
        reused = load_saved_unlearn_object(
            dataset=dataset_train_affine,
            load_dir=str(reuse_unlearn_dir),
        )
        dataset_unlearn = reused["dataset_unlearn"]
        dataset_remain = reused["dataset_remain"]
        unlearn_obj = reused["unlearn_object"]
        save_unlearn_object(unlearn_obj, str(result_dir))
    else:
        dataset_unlearn, dataset_remain, unlearn_obj = load_and_split_unlearn_datasets(
            cfg=cfg,
            model_type=common["model_type"],
            dataset=dataset_train_affine,
            index_mode=common["index_mode"],
            index_criteria=common["index_criteria"],
            unlearn_prop=float(cfg.unlearn_prop),
            selection_policy=common["selection_policy"],
            candidate_ratio=common["candidate_ratio"],
        )
        save_unlearn_object(unlearn_obj, str(result_dir))

    effective_policy = unlearn_obj["metadata"].get("selection_policy", "event_or_none")
    print("---------- index metadata ----------")
    print("selection_policy:", effective_policy)
    print("candidate_ratio:", unlearn_obj["metadata"].get("candidate_ratio"))
    print("candidate_no:", unlearn_obj["metadata"].get("candidate_no"))
    print("unlearn_no:", unlearn_obj["metadata"].get("unlearn_no"))
    print("num unlearn rows:", len(dataset_unlearn))
    print("num remain rows:", len(dataset_remain))

    dataset_collection = {
        "remain": dataset_remain,
        "unlearn": dataset_unlearn,
        "test": dataset_test_affine,
    }

    metrics_original = evaluate_all(model_ori, dataset_collection, cfg)

    parameter_direct, model_direct, direct_info = run_complete_unlearning_baseline(
        cfg=cfg,
        model_ori=model_ori,
        dataset_remain=dataset_remain,
        parameter_ori=parameter_ori,
        batch_size=common["batch_size"],
        train_loss=common["train_loss"],
    )
    metrics_direct = evaluate_all(model_direct, dataset_collection, cfg)

    shadow_bundle, shadow_weight_info = compute_shadow_price_sample_weights(
        cfg=cfg,
        model=model_ori,
        dataset_score=dataset_train_affine,
        result_dir=result_dir,
        bus_groups=ctx["bus_groups"],
        repair_criteria=repair_criteria,
    )
    safety_bundle, safety_weight_info = compute_safety_sample_gate(
        cfg=cfg,
        model=model_ori,
        dataset_score=dataset_train_affine,
        result_dir=result_dir,
    )

    if use_runtime:
        influence = compute_runtime_block_influence_for_eval(
            cfg=cfg,
            ctx=ctx,
            dataset_test_affine=dataset_test_affine,
            dataset_remain=dataset_remain,
            dataset_score=dataset_train_affine,
            unlearn_obj=unlearn_obj,
            result_dir=result_dir,
            repair_criteria=repair_criteria,
            client_score_weights=shadow_bundle.get("client_weight_matrix"),
            client_bus_score_weights=shadow_bundle.get("client_bus_weight_list"),
        )
        save_runtime_or_standard_influence_outputs(result_dir, influence)
    else:
        influence = compute_fed_influence(
            cfg=cfg,
            ctx=ctx,
            dataset_test_affine=dataset_test_affine,
            dataset_remain=dataset_remain,
            dataset_score=dataset_train_affine,
            repair_criteria=repair_criteria,
        )
        save_influence_outputs(result_dir, influence)

    print("---------- influence alignment ----------")
    print("grad:", influence["grad_alignment"])
    print("M:", influence["M_alignment"])
    print("score:", influence["score_alignment"])

    shadow_weights = shadow_bundle.get("sample_weights")
    safety_gate = safety_bundle.get("sample_gate")
    joint_score_mode = str(OmegaConf.select(cfg, "joint_score_mode", default="none"))
    joint_risk_mode = str(OmegaConf.select(cfg, "joint_risk_mode", default="stage2_ls"))
    joint_score_lambda = float(OmegaConf.select(cfg, "joint_score_lambda", default=0.0))
    use_joint_score = joint_score_mode not in ["", "none", "off", "false"]

    joint_sample_weights = None
    if shadow_weights is not None and safety_gate is not None and not use_joint_score:
        joint_sample_weights = np.asarray(shadow_weights, dtype=np.float64) * np.asarray(safety_gate, dtype=np.float64)
    elif shadow_weights is not None:
        joint_sample_weights = np.asarray(shadow_weights, dtype=np.float64)
    elif safety_gate is not None:
        joint_sample_weights = np.asarray(safety_gate, dtype=np.float64)

    if shadow_weights is not None:
        print("---------- shadow-weighted repair ----------")
        print(shadow_weight_info)
    if safety_gate is not None:
        print("---------- safety-gated repair ----------")
        print(safety_weight_info)

    score_for_repair_global = np.asarray(influence["scores_global"], dtype=np.float64)
    score_for_repair_fed = np.asarray(influence["scores_fed"], dtype=np.float64)
    joint_score_info = {
        "joint_score_mode": joint_score_mode,
        "joint_risk_mode": joint_risk_mode,
        "joint_score_lambda": joint_score_lambda,
    }
    if use_joint_score:
        raw_joint_risk = safety_bundle.get("sample_risk")
        if raw_joint_risk is None:
            raw_joint_risk = compute_safety_sample_risk(
                cfg=cfg,
                model=model_ori,
                dataset_score=dataset_train_affine,
                result_dir=result_dir,
                risk_mode=joint_risk_mode,
            )

        risk_global = compute_joint_risk_score(
            raw_risk=raw_joint_risk,
            score_reference=score_for_repair_global,
        )
        risk_fed = compute_joint_risk_score(
            raw_risk=raw_joint_risk,
            score_reference=score_for_repair_fed,
        )
        score_for_repair_global = score_for_repair_global - joint_score_lambda * risk_global["risk_score"]
        score_for_repair_fed = score_for_repair_fed - joint_score_lambda * risk_fed["risk_score"]

        np.save(result_dir / "joint_risk_score_global.npy", risk_global["risk_score"])
        np.save(result_dir / "joint_risk_score_fed.npy", risk_fed["risk_score"])
        pd.DataFrame({
            "sample_index": np.arange(risk_global["raw_risk"].size, dtype=int),
            "raw_risk": risk_global["raw_risk"],
            "risk_norm_global": risk_global["risk_norm"],
            "risk_norm_fed": risk_fed["risk_norm"],
            "joint_risk_score_global": risk_global["risk_score"],
            "joint_risk_score_fed": risk_fed["risk_score"],
        }).to_csv(result_dir / "joint_risk_score.csv", index=False)
        joint_score_info.update({
            "joint_risk_mean": risk_global["risk_mean"],
            "joint_score_scale_global": risk_global["score_scale"],
            "joint_score_scale_fed": risk_fed["score_scale"],
            "joint_risk_score_mean_global": float(np.mean(risk_global["risk_score"])),
            "joint_risk_score_mean_fed": float(np.mean(risk_fed["risk_score"])),
        })
        print("---------- joint-score repair ----------")
        print(joint_score_info)

    runtime_info = influence["extra_info"].get("runtime_info", {})
    topology_info = ctx.get("extra_info", {})

    base_info = {
        "model_type": common["model_type"],
        "eval_kind": "unchange_repair",
        "result_tag": str(OmegaConf.select(cfg, "result_tag", default="")),
        "fed_mode": common["fed_mode"],
        "feature_mode": affine_feature_mode,
        "affine_feature_mode": affine_feature_mode,
        "runtime_input_mode": runtime_input_mode,
        "requested_feature_mode": requested_feature_mode,
        "runtime_mode": runtime_mode,
        "fed_runtime": bool(use_runtime),
        "secure_agg_mode": secure_agg_mode,
        "index_mode": str(common["index_mode"]),
        "index_criteria": str(common["index_criteria"]),
        "criteria": repair_criteria,
        "repair_criteria": repair_criteria,
        "selection_policy": str(effective_policy),
        "reuse_unlearn_dir": "" if not reuse_unlearn_dir else str(Path(str(reuse_unlearn_dir)).resolve()),
        "unlearn_prop": float(cfg.unlearn_prop),
        "rho": float(common["rho"]),
        "block_damping": float(common["block_damping"]),
        "linf_constraint": float(common["linf_constraint"]),
        "num_bus_clients": int(common["num_bus_clients"]),
        "bus_groups": influence["bus_groups"],
        "topology_partition_enabled": topology_info.get("topology_partition_enabled", np.nan),
        "topology_fusion_enabled": topology_info.get("topology_fusion_enabled", np.nan),
        "topology_repair_regularization_enabled": topology_info.get(
            "topology_repair_regularization_enabled", np.nan
        ),
        "topology_encoder_propagation_enabled": topology_info.get(
            "topology_encoder_propagation_enabled", np.nan
        ),
        "topology_partition_mode": topology_info.get("topology_partition_mode", ""),
        "topology_tau": topology_info.get("topology_tau", np.nan),
        "topology_repair_rho": topology_info.get("topology_repair_rho", np.nan),
        "fusion_topology_alpha": topology_info.get("fusion_topology_alpha", np.nan),
        "server_feature_layout": topology_info.get("server_feature_layout", ""),
        "fusion_feature_layout": topology_info.get("fusion_feature_layout", ""),
        "encoder_self_weight": topology_info.get("encoder_self_weight", np.nan),
        "encoder_input_mode": topology_info.get("encoder_input_mode", ""),
        "encoder_topology_mode": topology_info.get("encoder_topology_mode", ""),
        "fusion_feature_dim": topology_info.get("fusion_feature_dim", np.nan),
        "shadow_weight_mode": shadow_weight_info.get("shadow_weight_mode", "none"),
        "shadow_weight_repair_criteria": shadow_weight_info.get("shadow_weight_repair_criteria", ""),
        "shadow_weight_min": shadow_weight_info.get("shadow_weight_min", np.nan),
        "shadow_weight_max": shadow_weight_info.get("shadow_weight_max", np.nan),
        "shadow_weight_tau": shadow_weight_info.get("shadow_weight_tau", np.nan),
        "shadow_weight_mean": shadow_weight_info.get("shadow_weight_final_mean", np.nan),
        "shadow_weight_std": shadow_weight_info.get("shadow_weight_final_std", np.nan),
        "safety_weight_mode": safety_weight_info.get("safety_weight_mode", "none"),
        "safety_weight_beta": safety_weight_info.get("safety_weight_beta", np.nan),
        "safety_weight_min": safety_weight_info.get("safety_weight_min", np.nan),
        "safety_weight_max": safety_weight_info.get("safety_weight_max", np.nan),
        "safety_gate_mean": safety_weight_info.get("safety_gate_mean", np.nan),
        "safety_gate_std": safety_weight_info.get("safety_gate_std", np.nan),
        "joint_score_mode": joint_score_info.get("joint_score_mode", "none"),
        "joint_risk_mode": joint_score_info.get("joint_risk_mode", ""),
        "joint_score_lambda": joint_score_info.get("joint_score_lambda", np.nan),
        "joint_risk_score_mean_global": joint_score_info.get("joint_risk_score_mean_global", np.nan),
        "joint_risk_score_mean_fed": joint_score_info.get("joint_risk_score_mean_fed", np.nan),
        "raw_parameter_diff": float(ctx["raw_parameter_diff"]),
        "grad_cosine_similarity": influence["grad_alignment"]["cosine_similarity"],
        "grad_relative_l2_error": influence["grad_alignment"]["relative_l2_error"],
        "M_cosine_similarity": influence["M_alignment"]["cosine_similarity"],
        "M_relative_l2_error": influence["M_alignment"]["relative_l2_error"],
        "score_cosine_similarity": influence["score_alignment"]["cosine_similarity"],
        "score_relative_l2_error": influence["score_alignment"]["relative_l2_error"],
        "H_local_condition": runtime_info.get("H_condition", np.nan),
        "H_local_solver": runtime_info.get("solver", ""),
        "runtime_payload_total_elements": runtime_info.get("payload_total_elements", np.nan),
        "global_cg_info": influence["hvp_info_global"].get("cg_info", np.nan),
        "global_cg_iterations": influence["hvp_info_global"].get("cg_iterations", np.nan),
        "global_cg_relative_residual": influence["hvp_info_global"].get("cg_relative_residual", np.nan),
        "fed_cg_info": influence["hvp_info_fed"].get("cg_info", np.nan),
        "fed_cg_iterations": influence["hvp_info_fed"].get("cg_iterations", np.nan),
        "fed_cg_relative_residual": influence["hvp_info_fed"].get("cg_relative_residual", np.nan),
    }

    rows = [
        build_summary_row("original", None, metrics_original, base_info),
        build_summary_row("direct_unlearn", None, metrics_direct, base_info),
    ]

    progress_csv_path = result_dir / "fed_unchange_repair_progress.csv"

    client_weight_scheme = str(shadow_weight_info.get("shadow_client_weight_scheme", ""))
    use_sample_weight_with_client_matrix = client_weight_scheme in ["per_sample_relative", "softmax_across_client"]

    repair_rows, repair_logs = run_fed_repair_grid(
        cfg=cfg,
        model_ori=model_ori,
        parameter_ori=parameter_ori,
        dataset_remain=dataset_remain,
        dataset_collection=dataset_collection,
        remain_index=unlearn_obj["remain_index"],
        scores_global=score_for_repair_global,
        scores_fed=score_for_repair_fed,
        base_info=base_info,
        batch_size=common["batch_size"],
        train_loss=common["train_loss"],
        linf_constraint=common["linf_constraint"],
        repair_criteria=repair_criteria,
        sample_weight_global=joint_sample_weights,
        sample_weight_fed=joint_sample_weights if (shadow_bundle.get("client_weight_matrix") is None or use_sample_weight_with_client_matrix) else None,
        progress_csv_path=progress_csv_path,
    )
    rows.extend(repair_rows)

    summary_df = pd.DataFrame(rows)
    summary_path = result_dir / "fed_unchange_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    log = {
        "context": describe_context(ctx),
        "base_info": base_info,
        "direct_unlearn_info": direct_info,
        "grad_alignment": influence["grad_alignment"],
        "M_alignment": influence["M_alignment"],
        "score_alignment": influence["score_alignment"],
        "hvp_info_global": influence["hvp_info_global"],
        "hvp_info_fed": influence["hvp_info_fed"],
        "score_info_global": influence["score_info_global"],
        "score_info_fed": influence["score_info_fed"],
        "extra_info": influence["extra_info"],
        "shadow_weight_info": shadow_weight_info,
        "safety_weight_info": safety_weight_info,
        "joint_score_info": joint_score_info,
        "repair_logs": repair_logs,
    }
    np.save(result_dir / "fed_unchange_log.npy", log, allow_pickle=True)

    print("\n========== Done ==========")
    print("summary:", summary_path)

    show_cols = [
        "method",
        "l1_constraint",
        "metric_mse_test",
        "metric_mape_test",
        "metric_cost_test",
        "metric_cost_unlearn",
        "feature_mode",
        "runtime_mode",
        "secure_agg_mode",
        "fed_runtime",
        "grad_cosine_similarity",
        "M_cosine_similarity",
        "score_cosine_similarity",
        "delta_metric_cost_test_fed_minus_global",
    ]
    show_cols = [c for c in show_cols if c in summary_df.columns]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
