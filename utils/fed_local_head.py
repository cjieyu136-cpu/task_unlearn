"""
utils/fed_local_head.py

Stage 3E prototype utility: client-local theta_k / local-output-head audit.

Purpose
-------
The previous bus-group Fed-TA-MU repair uses bus clients but still shares one
global theta. This module starts the next layer:

    client k owns local head parameters theta_k = (W_k, b_k)
    client k predicts only its bus group B_k
    CloudServer concatenates all bus-group predictions into full y_hat
    CloudServer computes g_y = d L_cost / d y_hat
    client k receives g_y[:, B_k]
    client k computes local VJP:
        grad_{theta_k} = J_{f_k(theta_k)}^T g_y[:, B_k]

This file implements an AUDIT ONLY:
    - extract the current topology-affine head into bus-local heads
    - verify local-head predictions reproduce the original global affine output
    - verify sum/assembly of local-head VJPs matches the global affine-head VJP

It does NOT yet run TA-MU repair with independent theta_k.
That should come only after this audit passes.

Assumptions
-----------
The current topology-affine head is affine on dataset_train_affine features:

    feature_dim = 64
    output_dim = 14

The model parameter count observed in prior runs is 910 = 64*14 + 14.
This utility defensively supports either Linear.weight shape:
    [output_dim, feature_dim] or [feature_dim, output_dim]
and a bias vector of length output_dim.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.fed_vjp_utils import (
    cloud_cost_gradient_scaled_for_dataset,
    model_output_scaled,
    alignment_metrics,
)
from utils.fed_bus_client import (
    parse_bus_groups,
    bus_groups_to_string,
)


# ---------------------------------------------------------------------
# Affine extraction
# ---------------------------------------------------------------------
def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def extract_affine_weight_bias(
    model: nn.Module,
    feature_dim: int,
    output_dim: int,
) -> Dict[str, Any]:
    """
    Extract affine head as canonical:

        y = x @ W_out_in.T + b

    where:
        W_out_in shape = [output_dim, feature_dim]
        b shape = [output_dim]

    Returns metadata about original parameter orientation.
    """
    params = list(model.parameters())
    if len(params) < 2:
        raise ValueError(
            f"Expected at least 2 parameters for affine head, got {len(params)}."
        )

    # Find a 2D weight and 1D bias.
    weight_param = None
    bias_param = None
    weight_index = None
    bias_index = None

    for i, p in enumerate(params):
        if p.ndim == 2 and p.numel() == feature_dim * output_dim and weight_param is None:
            weight_param = p
            weight_index = i
        elif p.ndim == 1 and p.numel() == output_dim and bias_param is None:
            bias_param = p
            bias_index = i

    if weight_param is None or bias_param is None:
        shapes = [tuple(p.shape) for p in params]
        raise ValueError(
            f"Could not identify affine weight/bias. "
            f"feature_dim={feature_dim}, output_dim={output_dim}, param_shapes={shapes}"
        )

    W = weight_param.detach().cpu().numpy().astype(np.float32)
    b = bias_param.detach().cpu().numpy().astype(np.float32)

    if W.shape == (output_dim, feature_dim):
        W_out_in = W.copy()
        orientation = "out_in"
    elif W.shape == (feature_dim, output_dim):
        W_out_in = W.T.copy()
        orientation = "in_out"
    else:
        raise ValueError(f"Unexpected weight shape {W.shape}")

    return {
        "W_out_in": W_out_in,
        "b": b.copy(),
        "orientation": orientation,
        "weight_index": weight_index,
        "bias_index": bias_index,
        "param_shapes": [tuple(p.shape) for p in params],
    }


def canonical_affine_forward(feature: torch.Tensor, W_out_in: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return feature @ W_out_in.t() + b


def flatten_canonical_grad(W_grad_out_in: np.ndarray, b_grad: np.ndarray) -> np.ndarray:
    """
    Canonical vector for comparing independent local heads:
        [W_out_in.flatten(), b.flatten()]
    """
    return np.concatenate([
        np.asarray(W_grad_out_in, dtype=float).reshape(-1),
        np.asarray(b_grad, dtype=float).reshape(-1),
    ])


# ---------------------------------------------------------------------
# Local heads
# ---------------------------------------------------------------------
class BusLocalHeadBank(nn.Module):
    """
    Client-local affine output heads.

    Each client owns:
        W_k: [num_buses_k, feature_dim]
        b_k: [num_buses_k]

    Forward returns full output [batch, output_dim] by placing each local
    prediction into the global bus positions.
    """

    def __init__(
        self,
        W_out_in: np.ndarray,
        b: np.ndarray,
        bus_groups: Sequence[np.ndarray],
    ):
        super().__init__()

        self.output_dim = int(W_out_in.shape[0])
        self.feature_dim = int(W_out_in.shape[1])
        self.bus_groups = [np.asarray(g, dtype=int).reshape(-1) for g in bus_groups]

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for buses in self.bus_groups:
            W_k = torch.tensor(W_out_in[buses, :], dtype=torch.float32)
            b_k = torch.tensor(b[buses], dtype=torch.float32)
            self.weights.append(nn.Parameter(W_k))
            self.biases.append(nn.Parameter(b_k))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = feature.new_zeros((feature.shape[0], self.output_dim))
        for buses, W_k, b_k in zip(self.bus_groups, self.weights, self.biases):
            pred_k = feature @ W_k.t() + b_k
            out[:, torch.tensor(buses, device=feature.device, dtype=torch.long)] = pred_k
        return out

    def forward_group(self, feature: torch.Tensor, client_id: int) -> torch.Tensor:
        W_k = self.weights[int(client_id)]
        b_k = self.biases[int(client_id)]
        return feature @ W_k.t() + b_k

    def assemble_canonical_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        W = np.zeros((self.output_dim, self.feature_dim), dtype=np.float32)
        b = np.zeros((self.output_dim,), dtype=np.float32)

        for buses, W_k, b_k in zip(self.bus_groups, self.weights, self.biases):
            W[buses, :] = W_k.detach().cpu().numpy()
            b[buses] = b_k.detach().cpu().numpy()

        return W, b


# ---------------------------------------------------------------------
# Prediction audit
# ---------------------------------------------------------------------
def collect_predictions(
    model: nn.Module,
    dataset: Any,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    dev = torch.device(device or "cpu")
    model = model.to(dev)
    model.eval()

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    preds = []
    with torch.no_grad():
        for feature, _target in loader:
            feature = feature.to(dev)
            pred = model_output_scaled(model, feature)
            preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(float)


def prediction_alignment(
    global_model: nn.Module,
    local_bank: BusLocalHeadBank,
    dataset: Any,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, float]:
    dev = torch.device(device or "cpu")
    local_bank = local_bank.to(dev)
    local_bank.eval()

    y_global = collect_predictions(global_model, dataset, batch_size=batch_size, device=device)

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    preds_local = []
    with torch.no_grad():
        for feature, _target in loader:
            feature = feature.to(dev)
            pred = local_bank(feature)
            preds_local.append(pred.detach().cpu().numpy())

    y_local = np.concatenate(preds_local, axis=0).astype(float)
    diff = y_local - y_global

    return {
        "prediction_rmse": float(np.sqrt(np.mean(diff ** 2))),
        "prediction_max_abs_diff": float(np.max(np.abs(diff))),
        "prediction_mean_abs_diff": float(np.mean(np.abs(diff))),
        "global_pred_norm": float(np.linalg.norm(y_global)),
        "local_pred_norm": float(np.linalg.norm(y_local)),
    }


# ---------------------------------------------------------------------
# VJP audit
# ---------------------------------------------------------------------
def global_affine_vjp_canonical(
    model: nn.Module,
    dataset: Any,
    gy_scaled: np.ndarray,
    feature_dim: int,
    output_dim: int,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute global affine-head VJP and return canonical [W_out_in, b] gradient.

    This uses the actual model parameters, then maps their gradient orientation
    into canonical W_out_in format.
    """
    dev = torch.device(device or "cpu")
    model = model.to(dev)
    model.eval()

    info = extract_affine_weight_bias(model, feature_dim=feature_dim, output_dim=output_dim)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    gy_scaled = np.asarray(gy_scaled, dtype=np.float32)

    start = 0
    for feature, _target in loader:
        bsz = feature.shape[0]
        end = start + bsz

        feature = feature.to(dev)
        gy_batch = torch.tensor(gy_scaled[start:end], dtype=torch.float32, device=dev)

        pred = model_output_scaled(model, feature)
        scalar = torch.sum(pred * gy_batch)

        grads = torch.autograd.grad(
            scalar,
            list(model.parameters()),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for p, g in zip(model.parameters(), grads):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad += g.detach()

        start = end

    params = list(model.parameters())
    W_grad_raw = params[info["weight_index"]].grad.detach().cpu().numpy().astype(np.float32)
    b_grad = params[info["bias_index"]].grad.detach().cpu().numpy().astype(np.float32)

    if info["orientation"] == "out_in":
        W_grad_out_in = W_grad_raw
    elif info["orientation"] == "in_out":
        W_grad_out_in = W_grad_raw.T
    else:
        raise ValueError(f"Unknown orientation: {info['orientation']}")

    vec = flatten_canonical_grad(W_grad_out_in, b_grad)

    return {
        "W_grad_out_in": W_grad_out_in,
        "b_grad": b_grad,
        "grad_vec": vec,
        "info": info,
    }


def local_head_vjp_canonical(
    local_bank: BusLocalHeadBank,
    dataset: Any,
    gy_scaled: np.ndarray,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute client-local theta_k VJPs and assemble them into canonical full
    [W_out_in, b] gradient for comparison.

    Each client only uses its bus-group gy slice.
    """
    dev = torch.device(device or "cpu")
    local_bank = local_bank.to(dev)
    local_bank.eval()

    for p in local_bank.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    gy_scaled = np.asarray(gy_scaled, dtype=np.float32)

    start = 0
    for feature, _target in loader:
        bsz = feature.shape[0]
        end = start + bsz

        feature = feature.to(dev)

        # Sum client-local scalars. This is equivalent to full output VJP
        # but uses only each client's output buses and local parameters.
        scalar_total = feature.new_tensor(0.0)
        for cid, buses in enumerate(local_bank.bus_groups):
            gy_batch = torch.tensor(gy_scaled[start:end][:, buses], dtype=torch.float32, device=dev)
            pred_k = local_bank.forward_group(feature, cid)
            scalar_total = scalar_total + torch.sum(pred_k * gy_batch)

        grads = torch.autograd.grad(
            scalar_total,
            list(local_bank.parameters()),
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for p, g in zip(local_bank.parameters(), grads):
            if g is None:
                continue
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad += g.detach()

        start = end

    W_grad = np.zeros((local_bank.output_dim, local_bank.feature_dim), dtype=np.float32)
    b_grad = np.zeros((local_bank.output_dim,), dtype=np.float32)

    per_client = []
    # Parameter order is weights[0], weights[1],..., biases interleaved in ParameterLists?
    # Safer: read .grad directly from corresponding weight/bias params.
    for cid, buses in enumerate(local_bank.bus_groups):
        Wg = local_bank.weights[cid].grad.detach().cpu().numpy().astype(np.float32)
        bg = local_bank.biases[cid].grad.detach().cpu().numpy().astype(np.float32)
        W_grad[buses, :] = Wg
        b_grad[buses] = bg
        per_client.append({
            "client_id": int(cid),
            "bus_indices": ",".join(str(int(x)) for x in buses),
            "num_buses": int(len(buses)),
            "theta_k_elements": int(Wg.size + bg.size),
            "grad_norm": float(np.sqrt(np.sum(Wg ** 2) + np.sum(bg ** 2))),
            "upload_gradient_elements": int(Wg.size + bg.size),
        })

    vec = flatten_canonical_grad(W_grad, b_grad)

    return {
        "W_grad_out_in": W_grad,
        "b_grad": b_grad,
        "grad_vec": vec,
        "per_client": per_client,
    }


# ---------------------------------------------------------------------
# Main audit entry
# ---------------------------------------------------------------------
def local_head_fed_vjp_audit(
    cfg: Any,
    global_model: nn.Module,
    dataset: Any,
    num_clients: int = 4,
    bus_groups: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Audit independent local output heads initialized from the global affine head.

    Returns prediction alignment and VJP alignment.
    """
    feature_dim = int(dataset.feature.shape[1])
    output_dim = int(dataset.target.shape[1])

    groups = parse_bus_groups(bus_groups=bus_groups, num_bus=output_dim, num_clients=num_clients)

    affine = extract_affine_weight_bias(
        global_model,
        feature_dim=feature_dim,
        output_dim=output_dim,
    )

    local_bank = BusLocalHeadBank(
        W_out_in=affine["W_out_in"],
        b=affine["b"],
        bus_groups=groups,
    )

    pred_metrics = prediction_alignment(
        global_model=global_model,
        local_bank=local_bank,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
    )

    cloud = cloud_cost_gradient_scaled_for_dataset(
        cfg=cfg,
        model=global_model,
        dataset=dataset,
        batch_size=batch_size,
        device=device,
    )
    gy = cloud["gy_scaled"]

    global_grad = global_affine_vjp_canonical(
        model=global_model,
        dataset=dataset,
        gy_scaled=gy,
        feature_dim=feature_dim,
        output_dim=output_dim,
        batch_size=batch_size,
        device=device,
    )

    local_grad = local_head_vjp_canonical(
        local_bank=local_bank,
        dataset=dataset,
        gy_scaled=gy,
        batch_size=batch_size,
        device=device,
    )

    vjp_metrics = alignment_metrics(global_grad["grad_vec"], local_grad["grad_vec"])

    return {
        "prediction_alignment": pred_metrics,
        "vjp_alignment": vjp_metrics,
        "affine_info": affine,
        "bus_groups": groups,
        "bus_groups_string": bus_groups_to_string(groups),
        "cloud_info": cloud["info"],
        "global_grad_vec": global_grad["grad_vec"],
        "local_grad_vec": local_grad["grad_vec"],
        "per_client": local_grad["per_client"],
        "info": {
            "feature_dim": feature_dim,
            "output_dim": output_dim,
            "num_clients": int(len(groups)),
            "bus_groups": bus_groups_to_string(groups),
            "global_theta_elements_canonical": int(global_grad["grad_vec"].size),
            "local_theta_elements_total": int(sum(x["theta_k_elements"] for x in local_grad["per_client"])),
        },
    }


# ---------------------------------------------------------------------
# Conversion between local-head canonical vector and repo model parameter order
# ---------------------------------------------------------------------
def canonical_grad_to_model_order(
    model: nn.Module,
    feature_dim: int,
    output_dim: int,
    W_grad_out_in: np.ndarray,
    b_grad: np.ndarray,
) -> np.ndarray:
    """
    Convert canonical local-head gradient into the actual model.parameters()
    flatten order used by repo-style reconstruct/IHVP utilities.

    canonical:
        W_grad_out_in shape [output_dim, feature_dim]
        b_grad shape [output_dim]

    model order may store weight as:
        [output_dim, feature_dim]  or  [feature_dim, output_dim]

    Any unused parameters are filled with zeros. For the current affine head,
    there should only be the affine weight and bias, total 910 elements.
    """
    info = extract_affine_weight_bias(model, feature_dim=feature_dim, output_dim=output_dim)

    W_grad_out_in = np.asarray(W_grad_out_in, dtype=np.float32)
    b_grad = np.asarray(b_grad, dtype=np.float32)

    if W_grad_out_in.shape != (output_dim, feature_dim):
        raise ValueError(
            f"W_grad_out_in shape {W_grad_out_in.shape} != {(output_dim, feature_dim)}"
        )
    if b_grad.shape != (output_dim,):
        raise ValueError(f"b_grad shape {b_grad.shape} != {(output_dim,)}")

    chunks = []
    for i, p in enumerate(model.parameters()):
        if i == info["weight_index"]:
            if info["orientation"] == "out_in":
                arr = W_grad_out_in
            elif info["orientation"] == "in_out":
                arr = W_grad_out_in.T
            else:
                raise ValueError(f"Unknown orientation: {info['orientation']}")
            chunks.append(arr.reshape(-1).astype(float))
        elif i == info["bias_index"]:
            chunks.append(b_grad.reshape(-1).astype(float))
        else:
            chunks.append(np.zeros(p.numel(), dtype=float))

    return np.concatenate(chunks).astype(float)


def canonical_vec_to_model_order(
    model: nn.Module,
    feature_dim: int,
    output_dim: int,
    canonical_vec: np.ndarray,
) -> np.ndarray:
    """
    Convert canonical [W_out_in.flatten(), b] vector to model flatten order.
    """
    canonical_vec = np.asarray(canonical_vec, dtype=np.float32).reshape(-1)
    expected = output_dim * feature_dim + output_dim
    if canonical_vec.size != expected:
        raise ValueError(f"canonical_vec has {canonical_vec.size} elements, expected {expected}")

    W = canonical_vec[: output_dim * feature_dim].reshape(output_dim, feature_dim)
    b = canonical_vec[output_dim * feature_dim :]
    return canonical_grad_to_model_order(
        model=model,
        feature_dim=feature_dim,
        output_dim=output_dim,
        W_grad_out_in=W,
        b_grad=b,
    )


def model_order_vec_to_canonical(
    model: nn.Module,
    feature_dim: int,
    output_dim: int,
    model_order_vec: np.ndarray,
) -> np.ndarray:
    """
    Convert flattened model.parameters() order into canonical
    [W_out_in.flatten(), b] order.
    """
    info = extract_affine_weight_bias(model, feature_dim=feature_dim, output_dim=output_dim)
    vec = np.asarray(model_order_vec, dtype=np.float64).reshape(-1)

    params = list(model.parameters())
    offset = 0
    weight_raw = None
    bias_raw = None

    for i, p in enumerate(params):
        chunk = vec[offset: offset + p.numel()].reshape(tuple(p.shape))
        if i == info["weight_index"]:
            weight_raw = chunk.copy()
        elif i == info["bias_index"]:
            bias_raw = chunk.reshape(-1).copy()
        offset += p.numel()

    if weight_raw is None or bias_raw is None:
        raise ValueError("Could not recover weight/bias from model_order_vec.")

    if info["orientation"] == "out_in":
        W_out_in = weight_raw
    elif info["orientation"] == "in_out":
        W_out_in = weight_raw.T
    else:
        raise ValueError(f"Unknown orientation: {info['orientation']}")

    return np.concatenate([W_out_in.reshape(-1), bias_raw.reshape(-1)]).astype(float)


def local_head_cost_gradient_model_order(
    cfg: Any,
    global_model: nn.Module,
    dataset: Any,
    num_clients: int = 4,
    bus_groups: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute client-local theta_k VJP gradient and return it in repo model
    parameter order.

    This is the bridge needed by run_fed_local_head_repair.py:
        local theta_k VJP -> canonical vector -> repo flatten_model order
    """
    audit = local_head_fed_vjp_audit(
        cfg=cfg,
        global_model=global_model,
        dataset=dataset,
        num_clients=num_clients,
        bus_groups=bus_groups,
        batch_size=batch_size,
        device=device,
    )

    feature_dim = int(audit["info"]["feature_dim"])
    output_dim = int(audit["info"]["output_dim"])

    local_model_order = canonical_vec_to_model_order(
        model=global_model,
        feature_dim=feature_dim,
        output_dim=output_dim,
        canonical_vec=audit["local_grad_vec"],
    )

    global_model_order = canonical_vec_to_model_order(
        model=global_model,
        feature_dim=feature_dim,
        output_dim=output_dim,
        canonical_vec=audit["global_grad_vec"],
    )

    model_order_alignment = alignment_metrics(global_model_order, local_model_order)

    audit["local_grad_model_order"] = local_model_order
    audit["global_grad_model_order"] = global_model_order
    audit["model_order_alignment"] = model_order_alignment

    return audit
