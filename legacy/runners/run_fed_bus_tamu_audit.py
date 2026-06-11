"""
run_fed_bus_tamu_audit.py

Stage 3D first bus/region-level Fed-TA-MU audit.

This is NOT a repair runner yet. It checks whether a bus-group client federation
can reproduce the centralized TA-MU-cost gradient.

Client definition
-----------------
Each client owns a group of output buses. The default 4-client split for IEEE-14
is contiguous:

    client 0: buses 0,1,2,3
    client 1: buses 4,5,6
    client 2: buses 7,8,9
    client 3: buses 10,11,12,13

Cloud:
    computes full OPF/cost gradient g_y = dL_cost/dy_hat

Client k:
    receives g_y[:, bus_group_k]
    computes local output VJP:
        J_{f[:, bus_group_k]}^T g_y[:, bus_group_k]

Server:
    sums client gradients and compares with centralized cost gradient.

Example
-------
python run_fed_bus_tamu_audit.py model=conv +rho=0.001 +num_bus_clients=4

Optional explicit bus groups:
python run_fed_bus_tamu_audit.py model=conv +rho=0.001 '+bus_groups=0,1,2,3|4,5,6|7,8,9|10,11,12,13'
"""

import os
import numpy as np
import pandas as pd
import hydra

from omegaconf import DictConfig, OmegaConf

from utils import return_dataset
from utils.optimization import Operator
from utils.topology import build_load_bus_adjacency, build_laplacian
from utils.topo_affine import return_topology_affine_model

from func_operation import (
    return_core_datasets,
    return_dataset_for_nn_affine,
)

from utils.reweight_utils import get_model_parameter_vector
from utils.fed_bus_client import compare_bus_fed_to_central_gradient


def format_float_for_path(x):
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def safe_tag(x):
    return str(x).replace("\\", "_").replace("/", "_").replace(".", "p").replace("|", "-").replace(",", "")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("========== Bus-group Fed-VJP Audit ==========")

    model_type = str(cfg.model.type)
    if "nn" not in model_type:
        raise ValueError("run_fed_bus_tamu_audit.py currently supports nn affine-head setting only.")

    rho = float(OmegaConf.select(cfg, "rho", default=1e-3))
    damping = float(OmegaConf.select(cfg, "damping", default=1e-8))
    batch_size = int(OmegaConf.select(cfg, "fed_bus_batch_size", default=128))
    num_bus_clients = int(OmegaConf.select(cfg, "num_bus_clients", default=4))
    bus_groups = OmegaConf.select(cfg, "bus_groups", default=None)

    print("model_type:", model_type)
    print("rho:", rho)
    print("damping:", damping)
    print("batch_size:", batch_size)
    print("num_bus_clients:", num_bus_clients)
    print("bus_groups override:", bus_groups)

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

    print("flatten_model parameter norm:", float(np.linalg.norm(parameter_ori)))
    print("raw parameter norm:", float(np.linalg.norm(parameter_ori_raw)))
    print("norm(flatten_model - raw_parameter):", float(np.linalg.norm(parameter_ori - parameter_ori_raw)))

    # ------------------------------------------------------------
    # Audit on test set: this is the TA-MU test criterion gradient.
    # ------------------------------------------------------------
    result = compare_bus_fed_to_central_gradient(
        cfg=cfg,
        model=model_ori,
        dataset=dataset_test_affine,
        num_clients=num_bus_clients,
        bus_groups=bus_groups,
        batch_size=batch_size,
    )

    print("\n---------- alignment ----------")
    print(result["alignment"])

    print("\n---------- bus client info ----------")
    for item in result["per_client"]:
        print(item)

    # ------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------
    bus_tag = safe_tag(result["bus_groups_string"])
    save_dir = os.path.join(
        str(cfg.simulation_dir),
        model_type,
        "top_fedtamu",
        "fed_bus_tamu_audit",
        f"clients_{num_bus_clients}_rho_{format_float_for_path(rho)}_groups_{bus_tag}",
    )
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "grad_centralized_cost.npy"), result["grad_centralized"])
    np.save(os.path.join(save_dir, "grad_fed_bus_cost.npy"), result["grad_fed_bus"])
    np.save(os.path.join(save_dir, "gy_test_scaled.npy"), result["gy_scaled"])

    client_df = pd.DataFrame(result["per_client"])
    client_df.to_csv(os.path.join(save_dir, "fed_bus_client_payload.csv"), index=False)

    summary = {
        "model_type": model_type,
        "rho": rho,
        "num_bus_clients": int(num_bus_clients),
        "bus_groups": result["bus_groups_string"],
        "test_grad_cosine_similarity": result["alignment"]["cosine_similarity"],
        "test_grad_relative_l2_error": result["alignment"]["relative_l2_error"],
        "test_grad_norm_ratio_fed_over_centralized": result["alignment"]["norm_ratio_fed_over_centralized"],
        "centralized_grad_norm": result["alignment"]["centralized_norm"],
        "fed_bus_grad_norm": result["alignment"]["fed_vjp_norm"],
        **result["fed_info"],
    }

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(save_dir, "fed_bus_tamu_audit_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n========== Saved ==========")
    print("save_dir:", save_dir)
    print("summary:", summary_path)
    print("client payload:", os.path.join(save_dir, "fed_bus_client_payload.csv"))


if __name__ == "__main__":
    main()
