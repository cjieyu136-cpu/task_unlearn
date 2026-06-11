"""
utils/fed_data_utils.py

Fed-TA-MU data/model preparation utilities.

This module mirrors the repository's data/model preparation responsibilities and
keeps them separate from influence and repair logic.

Responsibilities:
    - load repo datasets
    - build core/sensitive split and affine datasets
    - build topology-affine head
    - provide standard bus-client metadata

This file should not implement influence scores or repair.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import OmegaConf

from utils import return_dataset
from utils.optimization import Operator
from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.topo_affine import return_topology_affine_model
from utils.fed_topology_nn import build_topology_local_fusion_datasets, resolve_bus_groups_with_topology
from utils.fed_cache_utils import build_topology_fusion_cache_dir
from utils.funcs import ModelNNAffine

from func_operation import (
    return_core_datasets,
    return_dataset_for_nn_affine,
    return_nn_affine_model,
    return_nn_model,
)

from utils.reweight_utils import get_model_parameter_vector
from utils.fed_bus_client import parse_bus_groups, bus_groups_to_string


def format_float_for_path(x):
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def format_tag_for_path(tag: Optional[str]) -> str:
    if tag is None:
        return ""
    text = str(tag).strip()
    if text == "":
        return ""
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_"]:
            safe.append(ch)
        elif ch == ".":
            safe.append("p")
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def compact_tag_for_path(tag: Optional[str], max_len: int = 10) -> str:
    text = format_tag_for_path(tag)
    if text == "" or len(text) <= max_len:
        return text

    parts = [p for p in text.split("_") if p]
    if len(parts) >= 2:
        head = parts[0][: max_len - 1]
        tail = "".join(p[:1] for p in parts[1:])
        compact = f"{head}{tail}"[:max_len]
        if compact:
            return compact

    return text[:max_len]


def compact_index_mode_for_path(mode: Optional[str]) -> str:
    mapping = {
        "random": "rnd",
        "helpful": "hlp",
        "harmful": "hrm",
        "event_system": "evt",
        "event": "evt",
        "event_mask": "evm",
    }
    text = str(mode).lower()
    return mapping.get(text, text[:3] if len(text) > 3 else text)


def compact_criteria_for_path(criteria: Optional[str]) -> str:
    mapping = {
        "mse": "mse",
        "mape": "mpe",
        "cost": "cst",
    }
    text = str(criteria).lower()
    return mapping.get(text, text[:3] if len(text) > 3 else text)


def compact_runtime_mode_for_path(mode: Optional[str]) -> str:
    mapping = {
        "precomputed_local_cache": "pcache",
        "local_frozen_backbone": "lfb",
    }
    text = str(mode).lower()
    return mapping.get(text, text[:8] if len(text) > 8 else text)


def compact_feature_mode_for_path(mode: Optional[str]) -> str:
    mapping = {
        "precomputed_local_cache": "pcache",
        "topology_local_fusion": "tlf",
        "fusion_topology_only": "tlf",
        "local_mask_fusion": "tlf",
        "plain_repo_affine": "plain",
    }
    text = str(mode).lower()
    return mapping.get(text, text[:8] if len(text) > 8 else text)


def resolve_fed_common_cfg(cfg) -> Dict[str, Any]:
    """Collect common Fed-TA-MU config values without mutating cfg."""
    return {
        "model_type": str(cfg.model.type),
        "index_mode": OmegaConf.select(cfg, "index_mode", default=cfg.unlearn_mode),
        "index_criteria": str(OmegaConf.select(cfg, "index_criteria", default=cfg.criteria)),
        "selection_policy": OmegaConf.select(cfg, "selection_policy", default=None),
        "candidate_ratio": OmegaConf.select(cfg, "candidate_ratio", default=None),
        "rho": float(OmegaConf.select(cfg, "rho", default=1e-3)),
        "damping": float(OmegaConf.select(cfg, "damping", default=1e-8)),
        "block_damping": float(OmegaConf.select(cfg, "block_damping", default=1e-8)),
        "train_loss": str(OmegaConf.select(cfg, "train_loss", default="mse")),
        "linf_constraint": float(OmegaConf.select(cfg, "linf_constraint", default=1.0)),
        "batch_size": int(OmegaConf.select(cfg, "fed_batch_size", default=128)),
        "num_bus_clients": int(OmegaConf.select(cfg, "num_bus_clients", default=4)),
        "bus_groups_override": OmegaConf.select(cfg, "bus_groups", default=None),
        "fed_mode": str(OmegaConf.select(cfg, "fed_mode", default="block_Hk")),
    }


def resolve_feature_and_runtime_mode(cfg) -> Dict[str, str]:
    """
    Split topology feature construction from runtime execution mode.

    feature_mode:
        controls how affine repair features are constructed

    runtime_mode:
        controls whether clients consume precomputed affine features or compute
        backbone features from raw inputs during runtime simulation

    Backward compatibility:
        Old configs used:
            feature_mode=local_frozen_backbone
        which mixed runtime semantics into feature construction. We now map that
        legacy setting to:
            feature_mode=precomputed_local_cache
            runtime_mode=local_frozen_backbone
    """
    requested_feature_mode = str(OmegaConf.select(cfg, "feature_mode", default="precomputed_local_cache"))
    runtime_mode = OmegaConf.select(cfg, "runtime_mode", default=None)

    feature_mode = requested_feature_mode

    if runtime_mode is None:
        if requested_feature_mode == "local_frozen_backbone":
            runtime_mode = "local_frozen_backbone"
            feature_mode = "precomputed_local_cache"
        else:
            runtime_mode = "precomputed_local_cache"

    runtime_mode = str(runtime_mode)

    if feature_mode == "fusion_topology_only":
        feature_mode = "topology_local_fusion"
    elif feature_mode == "local_mask_fusion":
        feature_mode = "topology_local_fusion"

    valid_feature_modes = ["precomputed_local_cache", "topology_local_fusion"]
    if feature_mode not in valid_feature_modes:
        raise ValueError(
            "feature_mode must be one of "
            "precomputed_local_cache, topology_local_fusion, fusion_topology_only, or local_mask_fusion."
        )

    if runtime_mode not in ["precomputed_local_cache", "local_frozen_backbone"]:
        raise ValueError("runtime_mode must be precomputed_local_cache or local_frozen_backbone.")

    return {
        "requested_feature_mode": requested_feature_mode,
        "feature_mode": feature_mode,
        "runtime_mode": runtime_mode,
    }


def _legacy_topology_defaults(requested_feature_mode: str) -> Dict[str, bool]:
    mode = str(requested_feature_mode).lower()
    if mode in ["fusion_topology_only", "local_mask_fusion"]:
        return {
            "topology_partition_enabled": True,
            "topology_fusion_enabled": True,
            "topology_repair_regularization_enabled": True,
            "topology_encoder_propagation_enabled": False,
        }
    if mode == "topology_local_fusion":
        return {
            "topology_partition_enabled": True,
            "topology_fusion_enabled": True,
            "topology_repair_regularization_enabled": True,
            "topology_encoder_propagation_enabled": True,
        }
    return {
        "topology_partition_enabled": False,
        "topology_fusion_enabled": False,
        "topology_repair_regularization_enabled": True,
        "topology_encoder_propagation_enabled": False,
    }


def resolve_topology_controls(cfg, requested_feature_mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve the four topology dimensions plus server feature layout.

    The new explicit switches are:
        topology_partition_enabled
        topology_fusion_enabled
        topology_repair_regularization_enabled
        topology_encoder_propagation_enabled

    Backward compatibility:
        legacy feature_mode aliases still provide defaults, but explicit switch
        values always win.
    """
    requested_feature_mode = str(
        requested_feature_mode
        if requested_feature_mode is not None
        else OmegaConf.select(cfg, "feature_mode", default="precomputed_local_cache")
    )
    defaults = _legacy_topology_defaults(requested_feature_mode)

    partition_enabled = bool(
        OmegaConf.select(cfg, "topology_partition_enabled", default=defaults["topology_partition_enabled"])
    )
    fusion_enabled = bool(
        OmegaConf.select(cfg, "topology_fusion_enabled", default=defaults["topology_fusion_enabled"])
    )
    repair_enabled = bool(
        OmegaConf.select(
            cfg,
            "topology_repair_regularization_enabled",
            default=defaults["topology_repair_regularization_enabled"],
        )
    )
    encoder_enabled = bool(
        OmegaConf.select(
            cfg,
            "topology_encoder_propagation_enabled",
            default=defaults["topology_encoder_propagation_enabled"],
        )
    )

    topology_partition_mode = str(OmegaConf.select(cfg, "topology_partition_mode", default="topology"))
    topology_tau = float(OmegaConf.select(cfg, "topology_tau", default=1.0))
    encoder_self_weight = float(OmegaConf.select(cfg, "encoder_self_weight", default=3.0))
    encoder_input_mode = OmegaConf.select(cfg, "encoder_input_mode", default=None)
    if encoder_input_mode is None:
        encoder_topology_mode = str(OmegaConf.select(cfg, "encoder_topology_mode", default="topology"))
        if not encoder_enabled:
            encoder_topology_mode = str(
                OmegaConf.select(cfg, "encoder_disabled_mode", default="zero_fill_local")
            ).lower()
    else:
        encoder_topology_mode = str(encoder_input_mode).lower()

    fusion_alpha = float(OmegaConf.select(cfg, "fusion_topology_alpha", default=0.5))
    server_feature_layout = str(OmegaConf.select(cfg, "server_feature_layout", default=None) or "").strip()
    if server_feature_layout == "":
        server_feature_layout = str(OmegaConf.select(cfg, "fusion_feature_layout", default="concat_residual"))
    use_smoothing_residual = bool(OmegaConf.select(cfg, "fusion_use_smoothing_residual", default=True))
    normalize_fusion_feature = bool(OmegaConf.select(cfg, "fusion_normalize_feature", default=True))
    enable_fusion_cache = bool(OmegaConf.select(cfg, "enable_fusion_cache", default=True))

    return {
        "topology_partition_enabled": partition_enabled,
        "topology_fusion_enabled": fusion_enabled,
        "topology_repair_regularization_enabled": repair_enabled,
        "topology_encoder_propagation_enabled": encoder_enabled,
        "topology_partition_mode": topology_partition_mode,
        "topology_tau": topology_tau,
        "encoder_self_weight": encoder_self_weight,
        "encoder_topology_mode": encoder_topology_mode,
        "encoder_input_mode": encoder_topology_mode,
        "fusion_topology_alpha": fusion_alpha,
        "server_feature_layout": server_feature_layout,
        "fusion_use_smoothing_residual": use_smoothing_residual,
        "fusion_normalize_feature": normalize_fusion_feature,
        "enable_fusion_cache": enable_fusion_cache,
        "effective_feature_mode": "topology_local_fusion" if fusion_enabled else "precomputed_local_cache",
    }


def build_result_dir(
    cfg,
    model_type: str,
    fed_mode: str,
    eval_kind: str,
    index_mode: Optional[str] = None,
    index_criteria: Optional[str] = None,
) -> Path:
    """
    Standard Fed pipeline result directory.

    eval_kind:
        "unlearn" or "unchange"
    """
    rho = float(OmegaConf.select(cfg, "rho", default=1e-3))
    num_bus_clients = int(OmegaConf.select(cfg, "num_bus_clients", default=4))
    result_tag = compact_tag_for_path(OmegaConf.select(cfg, "result_tag", default=None))

    if eval_kind == "unlearn":
        if result_tag:
            short = (
                f"{compact_index_mode_for_path(index_mode)}_{index_criteria}"
                f"_p{format_float_for_path(float(cfg.unlearn_prop))}"
                f"_r{format_float_for_path(rho)}"
                f"_bc{num_bus_clients}"
            )
        else:
            short = (
                f"{str(index_mode).lower()}_{index_criteria}"
                f"_p{format_float_for_path(float(cfg.unlearn_prop))}"
                f"_r{format_float_for_path(rho)}"
                f"_bc{num_bus_clients}"
            )
    elif eval_kind == "unchange":
        short = f"unchange_r{format_float_for_path(rho)}_bc{num_bus_clients}"
    else:
        raise ValueError(f"Unsupported eval_kind: {eval_kind}")

    if result_tag:
        short = short.replace(str(index_criteria), compact_criteria_for_path(index_criteria))
        short = f"{short}_t_{result_tag}"

    return (
        Path(str(cfg.simulation_dir))
        / model_type
        / "top_fedtamu"
        / "fed_pipeline_refactor"
        / str(fed_mode)
        / eval_kind
        / short
    )


def prepare_repo_affine_context(cfg) -> Dict[str, Any]:
    """
    Load repository datasets and build the affine head.

    This mirrors the data/model setup used by the repo-style Stage-2 and Stage-3
    runners.
    """
    common = resolve_fed_common_cfg(cfg)
    model_type = common["model_type"]

    if "nn" not in model_type:
        raise ValueError("Fed-TA-MU refactor currently supports nn affine-head setting only.")

    dataset_train, dataset_test = return_dataset(cfg)

    dataset_core, dataset_sensitive = return_core_datasets(
        cfg,
        dataset_to_be_split=dataset_train,
    )

    affine_context_mode = str(OmegaConf.select(cfg, "affine_context_mode", default="topology_fusion")).lower()
    mode_info = resolve_feature_and_runtime_mode(cfg)
    feature_mode = mode_info["feature_mode"]
    requested_feature_mode = mode_info["requested_feature_mode"]
    runtime_mode = mode_info["runtime_mode"]
    topology_ctrl = resolve_topology_controls(cfg, requested_feature_mode=requested_feature_mode)
    topology_partition_mode = topology_ctrl["topology_partition_mode"]
    topology_tau = topology_ctrl["topology_tau"]
    encoder_self_weight = topology_ctrl["encoder_self_weight"]
    encoder_topology_mode = topology_ctrl["encoder_topology_mode"]
    fusion_alpha = topology_ctrl["fusion_topology_alpha"]
    server_feature_layout = topology_ctrl["server_feature_layout"]
    use_smoothing_residual = topology_ctrl["fusion_use_smoothing_residual"]
    normalize_fusion_feature = topology_ctrl["fusion_normalize_feature"]
    enable_fusion_cache = topology_ctrl["enable_fusion_cache"]
    feature_mode = topology_ctrl["effective_feature_mode"]

    operator = Operator(case_config=cfg.case)
    A_grid = build_load_bus_adjacency(operator)
    L_grid = build_laplacian(A_grid)

    extra_tables = {}
    extra_info = {
        "affine_context_mode": affine_context_mode,
        "feature_mode": feature_mode,
        "requested_feature_mode": requested_feature_mode,
        "runtime_mode": runtime_mode,
        "topology_partition_enabled": topology_ctrl["topology_partition_enabled"],
        "topology_fusion_enabled": topology_ctrl["topology_fusion_enabled"],
        "topology_repair_regularization_enabled": topology_ctrl["topology_repair_regularization_enabled"],
        "topology_encoder_propagation_enabled": topology_ctrl["topology_encoder_propagation_enabled"],
        "topology_partition_mode": topology_partition_mode,
        "topology_tau": topology_tau,
        "encoder_self_weight": encoder_self_weight,
        "encoder_topology_mode": encoder_topology_mode,
        "encoder_input_mode": topology_ctrl["encoder_input_mode"],
        "topology_repair_rho": float(common["rho"]) if topology_ctrl["topology_repair_regularization_enabled"] else 0.0,
        "fusion_topology_alpha": fusion_alpha,
        "server_feature_layout": server_feature_layout,
        "fusion_feature_layout": server_feature_layout,
        "fusion_use_smoothing_residual": use_smoothing_residual,
        "fusion_normalize_feature": normalize_fusion_feature,
    }

    if affine_context_mode == "plain_repo":
        dataset_train_affine, dataset_test_affine = return_dataset_for_nn_affine(
            cfg,
            dataset_sensitive,
            dataset_test,
        )
        output_dim = int(dataset_train_affine.target.shape[1])
        bus_groups = parse_bus_groups(
            bus_groups=common["bus_groups_override"],
            num_bus=output_dim,
            num_clients=common["num_bus_clients"],
        )
        bus_groups_string = bus_groups_to_string(bus_groups)
        parameter_ori_raw = np.asarray(return_nn_affine_model(dataset_train_affine), dtype=float).reshape(-1)
        model_ori = ModelNNAffine(
            parameter_ori_raw,
            no_out=output_dim,
        )
        model_ori.eval()
        extra_info.update({
            "feature_mode": "plain_repo_affine",
            "topology_partition_mode": "plain_repo",
            "encoder_topology_mode": "plain_repo",
            "fusion_topology_alpha": 0.0,
        })
    elif topology_ctrl["topology_fusion_enabled"]:
        effective_encoder_topology_mode = encoder_topology_mode
        frozen_backbone = return_nn_model(cfg, is_load=True, dataset="core")
        frozen_backbone.eval()
        cache_dir = None
        if enable_fusion_cache:
            cache_dir = build_topology_fusion_cache_dir(
                simulation_dir=str(cfg.simulation_dir),
                model_type=model_type,
                payload={
                    "model_type": model_type,
                    "feature_mode": feature_mode,
                    "topology_partition_enabled": topology_ctrl["topology_partition_enabled"],
                    "topology_fusion_enabled": topology_ctrl["topology_fusion_enabled"],
                    "topology_repair_regularization_enabled": topology_ctrl["topology_repair_regularization_enabled"],
                    "topology_encoder_propagation_enabled": topology_ctrl["topology_encoder_propagation_enabled"],
                    "num_bus_clients": common["num_bus_clients"],
                    "bus_groups_override": common["bus_groups_override"],
                    "partition_mode": topology_partition_mode,
                    "tau": topology_tau,
                    "encoder_self_weight": encoder_self_weight,
                    "encoder_topology_mode": effective_encoder_topology_mode,
                    "fusion_alpha": fusion_alpha,
                    "fusion_feature_layout": server_feature_layout,
                    "use_smoothing_residual": use_smoothing_residual,
                    "normalize_fusion_feature": normalize_fusion_feature,
                    "batch_size": common["batch_size"],
                    "sensitive_shape": tuple(dataset_sensitive.feature.shape),
                    "test_shape": tuple(dataset_test.feature.shape),
                },
            )

        fusion_ctx = build_topology_local_fusion_datasets(
            dataset_sensitive=dataset_sensitive,
            dataset_test=dataset_test,
            core_model=frozen_backbone,
            A_grid=A_grid,
            num_bus_clients=common["num_bus_clients"],
            bus_groups_override=common["bus_groups_override"],
            partition_mode=topology_partition_mode if topology_ctrl["topology_partition_enabled"] else "plain",
            tau=topology_tau,
            encoder_self_weight=encoder_self_weight,
            encoder_topology_mode=effective_encoder_topology_mode,
            topology_fusion_enabled=topology_ctrl["topology_fusion_enabled"],
            fusion_alpha=fusion_alpha,
            fusion_feature_layout=server_feature_layout,
            use_smoothing_residual=use_smoothing_residual,
            normalize_fusion_feature=normalize_fusion_feature,
            batch_size=common["batch_size"],
            device=str(OmegaConf.select(cfg, "device", default="cpu")),
            cache_dir=cache_dir,
        )
        dataset_train_affine = fusion_ctx.dataset_train_affine
        dataset_test_affine = fusion_ctx.dataset_test_affine
        bus_groups = fusion_ctx.bus_groups
        bus_groups_string = fusion_ctx.bus_groups_string
        extra_tables.update({
            "encoder_upload_summary": fusion_ctx.encoder_upload_summary,
            "client_embedding_summary": fusion_ctx.client_embedding_summary,
            "fusion_feature_summary": fusion_ctx.fusion_feature_summary,
        })
        extra_info.update({
            "bus_axis": int(fusion_ctx.bus_axis),
            "fusion_feature_dim": int(dataset_train_affine.feature.shape[1]),
            "fusion_cache_dir": "" if cache_dir is None else str(cache_dir),
            "effective_encoder_topology_mode": str(effective_encoder_topology_mode),
        })
        model_ori, parameter_ori_raw = return_topology_affine_model(
            dataset=dataset_train_affine,
            L_grid=L_grid,
            rho=common["rho"] if topology_ctrl["topology_repair_regularization_enabled"] else 0.0,
            damping=common["damping"],
        )
        model_ori.eval()
    else:
        dataset_train_affine, dataset_test_affine = return_dataset_for_nn_affine(
            cfg,
            dataset_sensitive,
            dataset_test,
        )
        output_dim = int(dataset_train_affine.target.shape[1])
        if topology_ctrl["topology_partition_enabled"]:
            bus_groups = resolve_bus_groups_with_topology(
                A_grid=A_grid,
                num_clients=common["num_bus_clients"],
                bus_groups_override=common["bus_groups_override"],
                partition_mode=topology_partition_mode,
            )
        else:
            bus_groups = parse_bus_groups(
                bus_groups=common["bus_groups_override"],
                num_bus=output_dim,
                num_clients=common["num_bus_clients"],
            )
        bus_groups_string = bus_groups_to_string(bus_groups)
        model_ori, parameter_ori_raw = return_topology_affine_model(
            dataset=dataset_train_affine,
            L_grid=L_grid,
            rho=common["rho"] if topology_ctrl["topology_repair_regularization_enabled"] else 0.0,
            damping=common["damping"],
        )
        model_ori.eval()

    parameter_ori = get_model_parameter_vector(model_ori)
    parameter_ori_raw = np.asarray(parameter_ori_raw, dtype=float).reshape(-1)

    output_dim = int(dataset_train_affine.target.shape[1])

    return {
        "common": common,
        "dataset_train": dataset_train,
        "dataset_test": dataset_test,
        "dataset_core": dataset_core,
        "dataset_sensitive": dataset_sensitive,
        "dataset_train_affine": dataset_train_affine,
        "dataset_test_affine": dataset_test_affine,
        "operator": operator,
        "A_grid": A_grid,
        "L_grid": L_grid,
        "model_ori": model_ori,
        "parameter_ori": parameter_ori,
        "parameter_ori_raw": parameter_ori_raw,
        "raw_parameter_diff": float(np.linalg.norm(parameter_ori - parameter_ori_raw)),
        "bus_groups": bus_groups,
        "bus_groups_string": bus_groups_string,
        "extra_tables": extra_tables,
        "extra_info": extra_info,
    }


def describe_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Small printable metadata block."""
    train_affine = ctx["dataset_train_affine"]
    test_affine = ctx["dataset_test_affine"]
    common = ctx["common"]

    return {
        "model_type": common["model_type"],
        "fed_mode": common["fed_mode"],
        "feature_mode": ctx.get("extra_info", {}).get("feature_mode", "precomputed_local_cache"),
        "runtime_mode": ctx.get("extra_info", {}).get("runtime_mode", "precomputed_local_cache"),
        "topology_partition_enabled": ctx.get("extra_info", {}).get("topology_partition_enabled", False),
        "topology_fusion_enabled": ctx.get("extra_info", {}).get("topology_fusion_enabled", False),
        "topology_repair_regularization_enabled": ctx.get("extra_info", {}).get(
            "topology_repair_regularization_enabled", True
        ),
        "topology_encoder_propagation_enabled": ctx.get("extra_info", {}).get(
            "topology_encoder_propagation_enabled", False
        ),
        "rho": common["rho"],
        "damping": common["damping"],
        "num_bus_clients": common["num_bus_clients"],
        "bus_groups": ctx["bus_groups_string"],
        "train_feature_shape": tuple(train_affine.feature.shape),
        "train_target_shape": tuple(train_affine.target.shape),
        "test_feature_shape": tuple(test_affine.feature.shape),
        "test_target_shape": tuple(test_affine.target.shape),
        "parameter_norm": float(np.linalg.norm(ctx["parameter_ori"])),
        "raw_parameter_norm": float(np.linalg.norm(ctx["parameter_ori_raw"])),
        "raw_parameter_diff": float(ctx["raw_parameter_diff"]),
    }
