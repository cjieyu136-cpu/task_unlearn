"""
utils/fed_bus_client.py

Stage 3D utility module: bus/region-level Fed-VJP audit.

Why this module exists
----------------------
The earlier sample-sharded simulator is a useful sanity check, but it is not a
realistic power-system federation. In an industrial power grid, an edge client
is more naturally a bus group, feeder, substation, region, or load aggregator.

This module therefore implements a bus-group federation audit:

    Client k:
        owns output buses B_k
        receives cloud-side g_y[:, B_k]
        computes local output-dimension VJP:
            grad_k = J_{f[:, B_k]}(theta)^T g_y[:, B_k]

    Cloud/server:
        computes full OPF cost gradient:
            g_y = d L_cost / d y_hat
        sends only the bus-slice g_y[:, B_k] to client k
        aggregates:
            grad_fed = sum_k grad_k

This is the correct next step after the sample-sharded audit.

Important boundary
------------------
This first bus-client version is an AUDIT, not a full repair runner yet.

It does NOT:
    - change the original repo TA-MU repair pipeline
    - introduce topology-sparse Hessian
    - create a new unlearn index
    - replace helpful/harmful/random/event logic
    - claim full production deployment

It checks whether bus/region-level cloud g_y + client VJP reproduces the
centralized cost gradient.

Repository / history alignment
------------------------------
It reuses:
    utils.fed_vjp_utils.cloud_cost_gradient_scaled_for_dataset
    utils.fed_vjp_utils.centralized_cost_gradient_repo_style
    utils.fed_vjp_utils.model_output_scaled
    utils.fed_vjp_utils.alignment_metrics

Those utilities were already aligned against the original repository's:
    Operator
    Stage_One_Layer
    Stage_Two_Layer
    SPO / MSE_COST cost behavior
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.fed_vjp_utils import (
    alignment_metrics,
    centralized_cost_gradient_repo_style,
    cloud_cost_gradient_scaled_for_dataset,
    model_output_scaled,
)


# ---------------------------------------------------------------------
# Bus grouping
# ---------------------------------------------------------------------
def make_contiguous_bus_groups(num_bus: int, num_clients: int) -> List[np.ndarray]:
    """
    Split bus indices into contiguous groups.

    Example:
        num_bus=14, num_clients=4 ->
            [0,1,2,3], [4,5,6], [7,8,9], [10,11,12,13]
    """
    groups = np.array_split(np.arange(int(num_bus), dtype=int), int(num_clients))
    return [g.astype(int) for g in groups if len(g) > 0]


def parse_bus_groups(bus_groups: Optional[str], num_bus: int, num_clients: int) -> List[np.ndarray]:
    """
    Parse bus group string.

    Format:
        "0,1,2,3|4,5,6|7,8,9|10,11,12,13"

    If None, use contiguous groups.
    """
    if bus_groups is None or str(bus_groups).strip() == "":
        return make_contiguous_bus_groups(num_bus=num_bus, num_clients=num_clients)

    groups = []
    for part in str(bus_groups).split("|"):
        buses = [int(x.strip()) for x in part.split(",") if x.strip() != ""]
        if not buses:
            continue
        groups.append(np.asarray(buses, dtype=int))

    if not groups:
        raise ValueError("bus_groups parsed to an empty list")

    flat = np.concatenate(groups)
    expected = set(range(int(num_bus)))
    actual = set(flat.tolist())

    if actual != expected:
        raise ValueError(
            f"bus_groups must cover exactly buses 0..{num_bus-1}. "
            f"Missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )

    if len(flat) != len(set(flat.tolist())):
        raise ValueError("bus_groups contains duplicate bus indices")

    return groups


def bus_groups_to_string(groups: Sequence[np.ndarray]) -> str:
    return "|".join(",".join(str(int(x)) for x in g) for g in groups)


# ---------------------------------------------------------------------
# Client bus-output VJP
# ---------------------------------------------------------------------
def client_bus_vjp_from_gy(
    model: torch.nn.Module,
    dataset: Any,
    gy_scaled: np.ndarray,
    bus_indices: Sequence[int],
    batch_size: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute local bus-output VJP for one bus/region client.

    This implements:
        grad_k = J_{f[:, B_k]}(theta)^T g_y[:, B_k]

    The model is the same architecture/parameter vector as the Stage-2
    topology-aware affine head. This audit decomposes the output-gradient
    contribution by bus group.
    """
    dev = torch.device(device or "cpu")
    model = model.to(dev)
    model.eval()

    buses = np.asarray(bus_indices, dtype=int).reshape(-1)
    gy_scaled = np.asarray(gy_scaled, dtype=np.float32)

    if gy_scaled.shape[0] != len(dataset):
        raise ValueError(f"gy length {gy_scaled.shape[0]} != dataset length {len(dataset)}")

    if np.max(buses) >= gy_scaled.shape[1] or np.min(buses) < 0:
        raise ValueError(f"Invalid bus index in {buses.tolist()} for gy shape {gy_scaled.shape}")

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)

    grad_chunks_sum = None
    start = 0

    for feature, _target in loader:
        bsz = feature.shape[0]
        end = start + bsz

        feature = feature.to(dev)
        gy_batch = torch.tensor(gy_scaled[start:end][:, buses], dtype=torch.float32, device=dev)

        pred = model_output_scaled(model, feature)
        pred_bus = pred[:, buses]

        scalar = torch.sum(pred_bus * gy_batch)

        grads = torch.autograd.grad(
            scalar,
            list(model.parameters()),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        flat = []
        for p, g in zip(model.parameters(), grads):
            if g is None:
                flat.append(torch.zeros_like(p).detach().reshape(-1).cpu().numpy())
            else:
                flat.append(g.detach().reshape(-1).cpu().numpy())

        grad_vec = np.concatenate(flat).astype(float)

        if grad_chunks_sum is None:
            grad_chunks_sum = grad_vec
        else:
            grad_chunks_sum += grad_vec

        start = end

    if grad_chunks_sum is None:
        return np.array([], dtype=float)

    return grad_chunks_sum


def bus_group_fed_vjp_gradient(
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    num_clients: int = 4,
    bus_groups: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Bus/region-level Fed-VJP gradient audit.

    Steps:
        1. Cloud computes full g_y using the OPF/cost layer.
        2. Cloud slices g_y by bus group.
        3. Each bus-client computes local VJP.
        4. Server aggregates local gradients.

    Returns aggregated gradient and per-client diagnostics.
    """
    # Cloud-side full g_y. This uses the same helper that was already checked
    # against centralized repo-style cost gradient.
    cloud = cloud_cost_gradient_scaled_for_dataset(
        cfg=cfg,
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
    )

    gy = cloud["gy_scaled"]
    num_bus = int(gy.shape[1])

    groups = parse_bus_groups(bus_groups=bus_groups, num_bus=num_bus, num_clients=num_clients)

    grad_sum = None
    per_client = []

    for cid, buses in enumerate(groups):
        grad_k = client_bus_vjp_from_gy(
            model=model,
            dataset=dataset,
            gy_scaled=gy,
            bus_indices=buses,
            batch_size=batch_size,
            device=device,
        )

        if grad_sum is None:
            grad_sum = grad_k.copy()
        else:
            grad_sum += grad_k

        gy_slice = gy[:, buses]
        per_client.append(
            {
                "client_id": int(cid),
                "bus_indices": ",".join(str(int(x)) for x in buses),
                "num_buses": int(len(buses)),
                "grad_norm": float(np.linalg.norm(grad_k)),
                "mean_abs_gy": float(np.mean(np.abs(gy_slice))),
                "max_abs_gy": float(np.max(np.abs(gy_slice))),
                "upload_prediction_elements": int(len(dataset) * len(buses)),
                "download_gy_elements": int(len(dataset) * len(buses)),
                "upload_gradient_elements": int(len(grad_k)),
            }
        )

    if grad_sum is None:
        grad_sum = np.array([], dtype=float)

    return {
        "grad_fed_bus": grad_sum.astype(float),
        "gy_scaled": gy,
        "cloud": cloud,
        "bus_groups": groups,
        "bus_groups_string": bus_groups_to_string(groups),
        "per_client": per_client,
        "info": {
            "num_clients": int(len(groups)),
            "num_bus": int(num_bus),
            "num_samples": int(len(dataset)),
            "bus_groups": bus_groups_to_string(groups),
            "grad_norm": float(np.linalg.norm(grad_sum)),
            "total_upload_prediction_elements": int(sum(x["upload_prediction_elements"] for x in per_client)),
            "total_download_gy_elements": int(sum(x["download_gy_elements"] for x in per_client)),
            "total_upload_gradient_elements": int(sum(x["upload_gradient_elements"] for x in per_client)),
        },
    }


def compare_bus_fed_to_central_gradient(
    cfg: Any,
    model: torch.nn.Module,
    dataset: Any,
    num_clients: int = 4,
    bus_groups: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare bus-group Fed-VJP gradient with centralized repo-style cost gradient.
    """
    fed = bus_group_fed_vjp_gradient(
        cfg=cfg,
        model=model,
        dataset=dataset,
        num_clients=num_clients,
        bus_groups=bus_groups,
        batch_size=batch_size,
        device=device,
    )

    central = centralized_cost_gradient_repo_style(
        cfg=cfg,
        model=model,
        dataset=dataset,
        batch_size=batch_size,
    )

    metrics = alignment_metrics(central, fed["grad_fed_bus"])

    return {
        "grad_centralized": central,
        "grad_fed_bus": fed["grad_fed_bus"],
        "gy_scaled": fed["gy_scaled"],
        "alignment": metrics,
        "fed_info": fed["info"],
        "per_client": fed["per_client"],
        "bus_groups_string": fed["bus_groups_string"],
    }
