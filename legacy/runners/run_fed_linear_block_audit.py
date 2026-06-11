from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from utils import return_dataset, flatten_model
from func_operation import return_trained_model, return_module, return_unlearn_datasets
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
from utils.fed_vjp_utils import alignment_metrics


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model_type = str(cfg.model.type)
    if model_type != "linear":
        raise ValueError("run_fed_linear_block_audit.py only supports model.type=linear")

    index_mode = str(OmegaConf.select(cfg, "index_mode", default=cfg.unlearn_mode))
    index_criteria = str(OmegaConf.select(cfg, "index_criteria", default=cfg.criteria))
    num_bus_clients = int(OmegaConf.select(cfg, "num_bus_clients", default=4))
    batch_size = int(cfg.data.batch_size_eval)

    dataset_train, dataset_test = return_dataset(cfg)
    influences = torch.from_numpy(np.load(cfg.model.influence_dir)).float()
    _, dataset_remain, unlearn_index, remain_index = return_unlearn_datasets(
        influences=influences,
        unlearn_prop=float(cfg.unlearn_prop),
        dataset_to_be_unlearn=dataset_train,
        mode=cfg.unlearn_mode,
        config=cfg,
    )

    model_ori = return_trained_model(cfg, model_type="linear", dataset_train=dataset_train, is_spo=False)
    parameter_ori = flatten_model(model_ori)

    bus_groups = default_bus_groups(num_buses=int(dataset_train.target.shape[1]), num_bus_clients=num_bus_clients)
    remain_partitions = build_linear_client_partitions(dataset_remain, bus_groups)
    total_row_count = flatten_linear_dataset(dataset_remain)[0].shape[0]

    gradient_batch_size = 1 if str(cfg.criteria) == "cost" else batch_size
    loader_train = DataLoader(dataset_train, batch_size=gradient_batch_size, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=gradient_batch_size, shuffle=False)
    model_test = return_trained_model(
        cfg,
        model_type="linear",
        dataset_train=dataset_train,
        is_spo=(str(cfg.criteria) == "cost"),
    )
    module_test = return_module(
        cfg,
        loss_type_dict={"train": "mse", "test": str(cfg.criteria)},
        loader_dict={"train": loader_train, "test": loader_test},
        model=model_test,
        method="cg",
    )
    grad_test = module_test.test_loss_grad(test_idxs=range(len(dataset_test))).numpy()

    H_fed = aggregate_linear_hessian(remain_partitions, total_row_count=total_row_count)
    M_fed = solve_linear_ihvp(H_fed, grad_test)
    M_global = solve_linear_ihvp(H_fed.copy(), grad_test)

    scores_global = compute_linear_sample_scores_global(dataset_train, parameter_ori, M_global)
    scores_fed = compute_linear_sample_scores_fed(dataset_train, bus_groups, parameter_ori, M_fed)

    grad_alignment = alignment_metrics(grad_test, grad_test)
    M_alignment = alignment_metrics(M_global, M_fed)
    score_alignment = alignment_metrics(scores_global, scores_fed)

    summary = pd.DataFrame(
        [
            {
                "index_mode": index_mode,
                "feature_mode": "raw_linear_feature",
                "fed_runtime": True,
                "unlearn_prop": float(cfg.unlearn_prop),
                "criteria": str(cfg.criteria),
                "rho": float(OmegaConf.select(cfg, "rho", default=0.0)),
                "num_bus_clients": num_bus_clients,
                "bus_groups": "|".join(",".join(str(int(v)) for v in g) for g in bus_groups),
                "grad_cosine_similarity": float(grad_alignment["cosine_similarity"]),
                "grad_relative_l2_error": float(grad_alignment["relative_l2_error"]),
                "M_cosine_similarity": float(M_alignment["cosine_similarity"]),
                "M_relative_l2_error": float(M_alignment["relative_l2_error"]),
                "score_cosine_similarity": float(score_alignment["cosine_similarity"]),
                "score_relative_l2_error": float(score_alignment["relative_l2_error"]),
                "num_unlearn": int(len(unlearn_index)),
                "num_remain": int(len(remain_index)),
            }
        ]
    )

    result_dir = (
        Path(str(cfg.simulation_dir))
        / model_type
        / "top_fedtamu"
        / "stage3h_runtime"
        / "linear_block_audit"
        / f"{index_mode}_{index_criteria}_p{str(cfg.unlearn_prop).replace('.', 'p')}_bc{num_bus_clients}"
    ).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(result_dir / "fed_linear_block_audit_summary.csv", index=False)
    linear_partition_summary(remain_partitions, total_row_count=total_row_count).to_csv(
        result_dir / "linear_client_partition_summary.csv",
        index=False,
    )
    linear_score_summary(dataset_train, bus_groups, parameter_ori, M_fed).to_csv(
        result_dir / "linear_score_client_summary.csv",
        index=False,
    )

    print("========== Linear Fed Block Audit ==========")
    print("result_dir:", str(result_dir))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
