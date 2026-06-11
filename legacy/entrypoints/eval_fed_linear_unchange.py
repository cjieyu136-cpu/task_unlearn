from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from utils import return_dataset, flatten_model
from func_operation import return_trained_model, return_module, return_unlearn_datasets
from utils.fed_data_utils import build_result_dir
from utils.index_utils import load_saved_unlearn_object, save_unlearn_object
from utils.fed_linear_runtime import (
    aggregate_linear_hessian,
    build_linear_client_partitions,
    compute_linear_sample_scores_fed,
    compute_linear_sample_scores_global,
    default_bus_groups,
    flatten_linear_dataset,
    linear_partition_summary,
    linear_score_summary,
    solve_linear_ihvp,
)
from utils.fed_repair_utils import (
    build_summary_row,
    evaluate_all,
    run_complete_unlearning_baseline,
    run_fed_repair_grid,
)
from utils.fed_vjp_utils import alignment_metrics


def _base_info(
    cfg,
    model_type,
    feature_mode,
    index_mode,
    index_criteria,
    repair_criteria,
    grad_align,
    M_align,
    score_align,
    bus_groups,
    reuse_unlearn_dir,
):
    return {
        "model_type": model_type,
        "eval_kind": "unchange_repair",
        "fed_mode": "shared_linear_hessian",
        "feature_mode": feature_mode,
        "fed_runtime": True,
        "secure_agg_mode": "none",
        "index_mode": index_mode,
        "index_criteria": index_criteria,
        "criteria": str(cfg.criteria),
        "repair_criteria": str(repair_criteria),
        "selection_policy": "repo_candidate_random",
        "reuse_unlearn_dir": "" if not reuse_unlearn_dir else str(Path(str(reuse_unlearn_dir)).resolve()),
        "unlearn_prop": float(cfg.unlearn_prop),
        "rho": float(OmegaConf.select(cfg, "rho", default=0.0)),
        "block_damping": 0.0,
        "linf_constraint": 1.0,
        "num_bus_clients": int(OmegaConf.select(cfg, "num_bus_clients", default=4)),
        "bus_groups": "|".join(",".join(str(int(v)) for v in g) for g in bus_groups),
        "raw_parameter_diff": 0.0,
        "grad_cosine_similarity": float(grad_align["cosine_similarity"]),
        "grad_relative_l2_error": float(grad_align["relative_l2_error"]),
        "M_cosine_similarity": float(M_align["cosine_similarity"]),
        "M_relative_l2_error": float(M_align["relative_l2_error"]),
        "score_cosine_similarity": float(score_align["cosine_similarity"]),
        "score_relative_l2_error": float(score_align["relative_l2_error"]),
        "H_local_condition": np.nan,
        "H_local_solver": "analytic",
        "runtime_payload_total_elements": np.nan,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model_type = str(cfg.model.type)
    if model_type != "linear":
        raise ValueError("eval_fed_linear_unchange.py only supports model.type=linear")

    feature_mode = "raw_linear_feature"
    repair_criteria = str(OmegaConf.select(cfg, "repair_criteria", default=cfg.criteria)).lower()
    reuse_unlearn_dir = OmegaConf.select(cfg, "reuse_unlearn_dir", default=None)
    index_mode = str(OmegaConf.select(cfg, "index_mode", default=cfg.unlearn_mode))
    inferred_index_criteria = ""
    influence_name = Path(str(cfg.model.influence_dir)).stem.lower()
    for candidate in ["mse", "mape", "cost"]:
        if candidate in influence_name:
            inferred_index_criteria = candidate
            break
    index_criteria = str(
        OmegaConf.select(
            cfg,
            "index_criteria",
            default=(inferred_index_criteria if inferred_index_criteria else cfg.criteria),
        )
    )
    batch_size = int(cfg.data.batch_size_eval)
    train_loss = str(cfg.model.train_loss)
    num_bus_clients = int(OmegaConf.select(cfg, "num_bus_clients", default=4))

    dataset_train, dataset_test = return_dataset(cfg)

    short_name = (
        f"{index_mode}_{index_criteria}"
        f"_p{str(cfg.unlearn_prop).replace('.', 'p')}"
        f"_bc{num_bus_clients}"
    )
    result_tag = str(OmegaConf.select(cfg, "result_tag", default="")).strip()
    if result_tag:
        short_name = f"{short_name}_t_{result_tag}"
    result_dir = (
        Path(str(cfg.simulation_dir))
        / model_type
        / "top_fedtamu"
        / "linear_runtime"
        / "unchange"
        / short_name
    ).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    if reuse_unlearn_dir:
        reused = load_saved_unlearn_object(
            dataset=dataset_train,
            load_dir=str(reuse_unlearn_dir),
        )
        dataset_unlearn = reused["dataset_unlearn"]
        dataset_remain = reused["dataset_remain"]
        unlearn_obj = reused["unlearn_object"]
        unlearn_index = np.asarray(unlearn_obj["unlearn_index"], dtype=int)
        remain_index = np.asarray(unlearn_obj["remain_index"], dtype=int)
        save_unlearn_object(unlearn_obj, str(result_dir))
    else:
        influences = torch.from_numpy(np.load(cfg.model.influence_dir)).float()
        dataset_unlearn, dataset_remain, unlearn_index, remain_index = return_unlearn_datasets(
            influences=influences,
            unlearn_prop=float(cfg.unlearn_prop),
            dataset_to_be_unlearn=dataset_train,
            mode=cfg.unlearn_mode,
            config=cfg,
        )
        unlearn_obj = {
            "unlearn_index": np.asarray(unlearn_index, dtype=int),
            "remain_index": np.asarray(remain_index, dtype=int),
            "index_mode": str(index_mode),
            "index_criteria": str(index_criteria),
            "unlearn_prop": float(cfg.unlearn_prop),
            "metadata": {
                "index_mode": str(index_mode),
                "index_criteria": str(index_criteria),
                "unlearn_prop": float(cfg.unlearn_prop),
                "selection_policy": "repo_candidate_random",
            },
        }
        save_unlearn_object(unlearn_obj, str(result_dir))

    model_ori = return_trained_model(cfg, model_type="linear", dataset_train=dataset_train, is_spo=False)
    parameter_ori = flatten_model(model_ori)

    dataset_collection = {
        "remain": dataset_remain,
        "unlearn": dataset_unlearn,
        "test": dataset_test,
    }

    metrics_original = evaluate_all(model_ori, dataset_collection, cfg)
    parameter_direct, model_direct, direct_info = run_complete_unlearning_baseline(
        cfg=cfg,
        model_ori=model_ori,
        dataset_remain=dataset_remain,
        parameter_ori=parameter_ori,
        batch_size=batch_size,
        train_loss=train_loss,
    )
    metrics_direct = evaluate_all(model_direct, dataset_collection, cfg)

    bus_groups = default_bus_groups(num_buses=int(dataset_train.target.shape[1]), num_bus_clients=num_bus_clients)
    remain_partitions = build_linear_client_partitions(dataset_remain, bus_groups)
    total_row_count = flatten_linear_dataset(dataset_remain)[0].shape[0]

    gradient_batch_size = 1 if repair_criteria == "cost" else batch_size
    loader_train = DataLoader(dataset_train, batch_size=gradient_batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=gradient_batch_size, shuffle=False)
    model_test = return_trained_model(
        cfg,
        model_type="linear",
        dataset_train=dataset_train,
        is_spo=(repair_criteria == "cost"),
    )
    module_test = return_module(
        cfg,
        loss_type_dict={"train": "mse", "test": repair_criteria},
        loader_dict={"train": loader_train, "test": loader_test},
        model=model_test,
        method="cg",
    )
    grad_test = module_test.test_loss_grad(test_idxs=range(len(dataset_test))).numpy()

    H_fed = aggregate_linear_hessian(remain_partitions, total_row_count=total_row_count)
    M_fed = solve_linear_ihvp(H_fed, grad_test)

    H_global = H_fed.copy()
    M_global = solve_linear_ihvp(H_global, grad_test)

    scores_global = compute_linear_sample_scores_global(dataset_train, parameter_ori, M_global)
    scores_fed = compute_linear_sample_scores_fed(dataset_train, bus_groups, parameter_ori, M_fed)

    grad_alignment = alignment_metrics(grad_test, grad_test)
    M_alignment = alignment_metrics(M_global, M_fed)
    score_alignment = alignment_metrics(scores_global, scores_fed)

    client_partition_df = linear_partition_summary(remain_partitions, total_row_count=total_row_count)
    client_score_df = linear_score_summary(dataset_train, bus_groups, parameter_ori, M_fed)
    client_partition_df.to_csv(result_dir / "runtime_linear_client_partition_summary.csv", index=False)
    client_score_df.to_csv(result_dir / "runtime_linear_score_client_summary.csv", index=False)
    np.save(result_dir / "grad_global.npy", grad_test)
    np.save(result_dir / "grad_fed.npy", grad_test)
    np.save(result_dir / "M_global.npy", M_global)
    np.save(result_dir / "M_fed.npy", M_fed)
    np.save(result_dir / "scores_global.npy", scores_global)
    np.save(result_dir / "scores_fed.npy", scores_fed)
    np.save(result_dir / "unlearn_index.npy", np.asarray(unlearn_index))
    np.save(result_dir / "remain_index.npy", np.asarray(remain_index))

    base_info = _base_info(
        cfg=cfg,
        model_type=model_type,
        feature_mode=feature_mode,
        index_mode=index_mode,
        index_criteria=index_criteria,
        repair_criteria=repair_criteria,
        grad_align=grad_alignment,
        M_align=M_alignment,
        score_align=score_alignment,
        bus_groups=bus_groups,
        reuse_unlearn_dir=reuse_unlearn_dir,
    )

    rows = [
        build_summary_row("original", None, metrics_original, base_info),
        build_summary_row("direct_unlearn", None, metrics_direct, base_info, repair_info=direct_info),
    ]

    repair_rows, repair_logs = run_fed_repair_grid(
        cfg=cfg,
        model_ori=model_ori,
        parameter_ori=parameter_ori,
        dataset_remain=dataset_remain,
        dataset_collection=dataset_collection,
        remain_index=remain_index,
        scores_global=scores_global,
        scores_fed=scores_fed,
        base_info=base_info,
        batch_size=batch_size,
        train_loss=train_loss,
        linf_constraint=1.0,
    )
    rows.extend(repair_rows)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(result_dir / "fed_unchange_summary.csv", index=False)
    np.save(
        result_dir / "fed_unchange_log.npy",
        {
            "summary_rows": rows,
            "repair_logs": repair_logs,
            "grad_alignment": grad_alignment,
            "M_alignment": M_alignment,
            "score_alignment": score_alignment,
        },
        allow_pickle=True,
    )

    print("========== Linear Fed Unchange Repair ==========")
    print("result_dir:", str(result_dir))
    print("grad alignment:", grad_alignment)
    print("M alignment:", M_alignment)
    print("score alignment:", score_alignment)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
