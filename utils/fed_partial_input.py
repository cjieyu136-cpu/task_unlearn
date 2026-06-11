"""
utils/fed_partial_input.py

Stage 3H H2-3 utilities for auditing partial/local raw input.

Goal
----
Test whether each bus/region client can run the existing shared frozen backbone
using only local raw input.

Important
---------
This utility does NOT assume that partial raw input is valid for the existing
repo backbone. It audits that assumption.

If the original backbone was trained on full-system input, masking non-local
raw dimensions will usually change h(x). In that case, "each region client owns
only local raw input" requires a regional/local backbone architecture or
retraining, not just a runtime wrapper.
"""

from typing import Sequence, Optional, Dict, Any
import numpy as np
import torch


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def infer_bus_axis(raw_feature, output_dim: int) -> Optional[int]:
    """
    Infer which axis of raw_feature corresponds to bus dimension.

    Returns:
        axis index if exactly one non-batch axis has length output_dim,
        otherwise None.
    """
    shape = tuple(raw_feature.shape)
    candidate_axes = []
    for axis, size in enumerate(shape):
        if axis == 0:
            continue
        if int(size) == int(output_dim):
            candidate_axes.append(axis)

    if len(candidate_axes) == 1:
        return candidate_axes[0]
    return None


def make_bus_masked_raw_feature(raw_feature, bus_indices: Sequence[int], output_dim: int, bus_axis: Optional[int] = None):
    """
    Zero out all non-local bus entries along bus_axis.

    Supports numpy arrays and torch tensors.

    This preserves original input shape so it can be passed into the existing
    frozen backbone. It is an audit proxy for local-only raw input, not a new
    regional model.
    """
    is_torch = isinstance(raw_feature, torch.Tensor)
    x = raw_feature.clone() if is_torch else np.array(raw_feature, copy=True)

    if bus_axis is None:
        bus_axis = infer_bus_axis(x, output_dim)

    if bus_axis is None:
        raise ValueError(
            "Could not infer bus axis. No unique non-batch axis equals output_dim. "
            "Pass +partial_input_bus_axis=<axis> explicitly after inspecting raw feature shape."
        )

    slicer = [slice(None)] * len(tuple(x.shape))
    keep = np.asarray(bus_indices, dtype=int)

    # Build zero mask on bus axis.
    if is_torch:
        mask_shape = [1] * len(tuple(x.shape))
        mask_shape[bus_axis] = int(x.shape[bus_axis])
        mask = torch.zeros(mask_shape, dtype=x.dtype, device=x.device)
        idx = torch.tensor(keep, dtype=torch.long, device=x.device)
        mask.index_fill_(bus_axis, idx, 1.0)
        return x * mask, bus_axis
    else:
        mask_shape = [1] * len(tuple(x.shape))
        mask_shape[bus_axis] = int(x.shape[bus_axis])
        mask = np.zeros(mask_shape, dtype=x.dtype)
        mask_index = [slice(None)] * len(mask_shape)
        mask_index[bus_axis] = keep
        mask[tuple(mask_index)] = 1
        return x * mask, bus_axis


def alignment(a, b) -> Dict[str, Any]:
    a = to_numpy(a).astype(np.float64).reshape(-1)
    b = to_numpy(b).astype(np.float64).reshape(-1)
    diff = a - b
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "relative_l2_error": float(np.linalg.norm(diff) / (np.linalg.norm(a) + 1e-12)),
        "cosine_similarity": float(np.dot(a, b) / denom) if denom > 0 else np.nan,
    }
