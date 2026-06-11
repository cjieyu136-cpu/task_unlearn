"""
utils/fed_client_runtime.py

Stage 3H FedClient runtime abstraction.

Supported feature modes:
1. precomputed_local_cache
   Client owns local affine features X = h(x). This is H2-1.

2. local_frozen_backbone
   Client owns raw input tensors and a frozen core NN backbone. The client
   locally computes X = h(x) before prediction / VJP / H_k / M_k / score.

This is single-machine client/server simulation, not encryption or networking.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Any, List

import numpy as np
import torch


@dataclass
class FedClientMessage:
    client_id: int
    message_type: str
    payload_shape: str
    num_elements: int
    metadata: Dict[str, Any]


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.tensor(x).float().to(device)


def _slice_rows(arr: np.ndarray, indices: Optional[np.ndarray]) -> np.ndarray:
    if indices is None:
        return arr
    return arr[np.asarray(indices, dtype=int)]


@torch.no_grad()
def compute_backbone_features(core_model, raw_feature, batch_size: int = 128, device: str = "cpu") -> np.ndarray:
    """
    Run frozen core model locally and return h(x).

    Repository NN_CONV / MLPMixer forward convention:
        out = model(x)
        out[0] is feature
        out[1] is forecast
    """
    core_model = core_model.to(device)
    core_model.eval()

    x = _to_tensor(raw_feature, device=device)

    feats = []
    for start in range(0, len(x), int(batch_size)):
        xb = x[start:start + int(batch_size)]
        out = core_model(xb)
        if isinstance(out, (tuple, list)):
            feat = out[0]
        else:
            raise RuntimeError(
                "core_model forward did not return (feature, forecast). "
                "Please check the selected model architecture."
            )
        feats.append(feat.detach().cpu().numpy())

    return np.concatenate(feats, axis=0).astype(np.float64)


class FedClient:
    """
    Bus-output Fed-TA-MU client.

    Local prediction:
        y_k = X @ W_k.T + b_k

    Owns:
        X = local feature cache, or raw input + frozen backbone for generating X
        Y[:, B_k]
        W_k, b_k
        bus indices B_k
    """

    def __init__(
        self,
        client_id: int,
        bus_indices: Sequence[int],
        target_slice: np.ndarray,
        W_k: np.ndarray,
        b_k: np.ndarray,
        output_dim: int,
        feature_mode: str = "precomputed_local_cache",
        feature_cache: Optional[np.ndarray] = None,
        raw_feature: Optional[Any] = None,
        frozen_backbone: Optional[torch.nn.Module] = None,
        feature_batch_size: int = 128,
        device: str = "cpu",
    ):
        self.client_id = int(client_id)
        self.bus_indices = np.asarray(bus_indices, dtype=int).reshape(-1)

        self.feature_mode = str(feature_mode)
        self._feature_cache = None if feature_cache is None else np.asarray(feature_cache, dtype=np.float64)
        self._raw_feature = raw_feature
        self._frozen_backbone = frozen_backbone
        self.feature_batch_size = int(feature_batch_size)
        self.device = str(device)

        self._Y = np.asarray(target_slice, dtype=np.float64)
        self.W_k = np.asarray(W_k, dtype=np.float64)
        self.b_k = np.asarray(b_k, dtype=np.float64).reshape(-1)
        self.output_dim = int(output_dim)

        if self._Y.shape[1] != len(self.bus_indices):
            raise ValueError("target_slice width must equal number of bus_indices")
        if self.b_k.shape != (len(self.bus_indices),):
            raise ValueError("b_k shape incompatible with bus group")

        X = self.features
        self.feature_dim = int(X.shape[1])

        if self.W_k.shape != (len(self.bus_indices), self.feature_dim):
            raise ValueError(
                f"W_k shape {self.W_k.shape} incompatible with "
                f"({len(self.bus_indices)}, {self.feature_dim})"
            )
        if X.shape[0] != self._Y.shape[0]:
            raise ValueError(f"feature rows {X.shape[0]} must equal target rows {self._Y.shape[0]}")

        self._gy_by_split: Dict[str, np.ndarray] = {}

    @property
    def features(self) -> np.ndarray:
        """
        Return local feature matrix X = h(x).

        In local_frozen_backbone mode, features are computed locally once and cached.
        """
        if self._feature_cache is not None:
            return self._feature_cache

        if self.feature_mode == "local_frozen_backbone":
            if self._raw_feature is None or self._frozen_backbone is None:
                raise RuntimeError("local_frozen_backbone mode requires raw_feature and frozen_backbone")
            self._feature_cache = compute_backbone_features(
                core_model=self._frozen_backbone,
                raw_feature=self._raw_feature,
                batch_size=self.feature_batch_size,
                device=self.device,
            )
            return self._feature_cache

        raise RuntimeError(f"No feature cache available for feature_mode={self.feature_mode}")

    def compute_local_features(self, force_recompute: bool = False) -> Dict[str, Any]:
        if force_recompute:
            self._feature_cache = None
        X = self.features
        return {
            "feature_shape": tuple(X.shape),
            "feature_norm": float(np.linalg.norm(X)),
            "message": self._message(
                "local_feature_ready",
                np.asarray([X.shape[0], X.shape[1]], dtype=np.int64),
                {
                    **self.local_metadata(include_private_shapes=True),
                    "feature_mode": self.feature_mode,
                    "feature_norm": float(np.linalg.norm(X)),
                },
            ),
        }

    def local_metadata(self, include_private_shapes: bool = False) -> Dict[str, Any]:
        X = self.features
        meta = {
            "client_id": self.client_id,
            "bus_indices": ",".join(str(int(x)) for x in self.bus_indices),
            "num_buses": int(len(self.bus_indices)),
            "num_samples": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "theta_k_elements": int(self.W_k.size + self.b_k.size),
            "feature_mode": self.feature_mode,
        }
        if include_private_shapes:
            meta["target_slice_shape"] = str(tuple(self._Y.shape))
            meta["has_raw_feature"] = bool(self._raw_feature is not None)
            meta["has_frozen_backbone"] = bool(self._frozen_backbone is not None)
        return meta

    def _message(self, message_type: str, arr: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        return FedClientMessage(
            client_id=self.client_id,
            message_type=message_type,
            payload_shape=str(tuple(arr.shape)),
            num_elements=int(arr.size),
            metadata=metadata or {},
        )

    def predict_slice(self, indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        X = _slice_rows(self.features, indices)
        pred = X @ self.W_k.T + self.b_k.reshape(1, -1)
        return {
            "prediction": pred,
            "message": self._message(
                "prediction_slice",
                pred,
                {"bus_indices": self.local_metadata()["bus_indices"], "feature_mode": self.feature_mode},
            ),
        }

    def receive_gy_slice(self, gy_slice: np.ndarray, split_name: str = "test") -> FedClientMessage:
        gy_slice = np.asarray(gy_slice, dtype=np.float64)
        if gy_slice.shape[1] != len(self.bus_indices):
            raise ValueError("gy_slice width must equal number of client buses")
        self._gy_by_split[str(split_name)] = gy_slice
        return self._message(
            "receive_gy_slice",
            gy_slice,
            {
                "split_name": split_name,
                "bus_indices": self.local_metadata()["bus_indices"],
                "feature_mode": self.feature_mode,
            },
        )

    def compute_criterion_gradient(self, indices: Optional[np.ndarray] = None, split_name: str = "test") -> Dict[str, Any]:
        if str(split_name) not in self._gy_by_split:
            raise RuntimeError(f"No gy received for split_name={split_name}")

        X = _slice_rows(self.features, indices)
        gy = self._gy_by_split[str(split_name)]
        if indices is not None:
            gy = gy[np.asarray(indices, dtype=int)]

        W_grad = gy.T @ X
        b_grad = np.sum(gy, axis=0)
        grad_vec = np.concatenate([W_grad.reshape(-1), b_grad.reshape(-1)])

        return {
            "W_grad": W_grad,
            "b_grad": b_grad,
            "grad_vec": grad_vec,
            "message": self._message("criterion_gradient", grad_vec, self.local_metadata()),
        }

    def compute_cost_gradient(self, indices: Optional[np.ndarray] = None, split_name: str = "test") -> Dict[str, Any]:
        # Backward-compatible alias for existing cost runtime path.
        out = self.compute_criterion_gradient(indices=indices, split_name=split_name)
        out["message"] = self._message("cost_gradient", out["grad_vec"], self.local_metadata())
        return out

    def local_mse_hessian_augmented(self, indices: Optional[np.ndarray] = None, damping: float = 1e-8) -> np.ndarray:
        X = _slice_rows(self.features, indices)
        N = int(X.shape[0])
        X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float64)], axis=1)

        H = (2.0 / float(N * self.output_dim)) * (X_aug.T @ X_aug)
        if damping is not None and float(damping) > 0:
            H = H + float(damping) * np.eye(H.shape[0], dtype=np.float64)
        return H

    def compute_block_ihvp(
        self,
        W_grad: np.ndarray,
        b_grad: np.ndarray,
        indices: Optional[np.ndarray] = None,
        damping: float = 1e-8,
    ) -> Dict[str, Any]:
        W_grad = np.asarray(W_grad, dtype=np.float64)
        b_grad = np.asarray(b_grad, dtype=np.float64).reshape(-1)

        H = self.local_mse_hessian_augmented(indices=indices, damping=damping)

        W_M = np.zeros_like(W_grad, dtype=np.float64)
        b_M = np.zeros_like(b_grad, dtype=np.float64)

        solver = "solve"
        for j in range(len(self.bus_indices)):
            g_aug = np.concatenate([W_grad[j, :], np.asarray([b_grad[j]], dtype=np.float64)])
            try:
                m_aug = -np.linalg.solve(H, g_aug)
            except np.linalg.LinAlgError:
                m_aug = -np.linalg.lstsq(H, g_aug, rcond=None)[0]
                solver = "lstsq"
            W_M[j, :] = m_aug[:-1]
            b_M[j] = m_aug[-1]

        M_vec = np.concatenate([W_M.reshape(-1), b_M.reshape(-1)])

        return {
            "W_M": W_M,
            "b_M": b_M,
            "M_vec": M_vec,
            "H_condition": float(np.linalg.cond(H)),
            "solver": solver,
            "message": self._message(
                "block_ihvp",
                M_vec,
                {**self.local_metadata(), "H_condition": float(np.linalg.cond(H)), "solver": solver},
            ),
        }

    def compute_score_contribution(
        self,
        W_M: np.ndarray,
        b_M: np.ndarray,
        indices: Optional[np.ndarray] = None,
        normalize_by_n: bool = True,
        bus_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        X = _slice_rows(self.features, indices)
        Y = _slice_rows(self._Y, indices)

        W_M = np.asarray(W_M, dtype=np.float64)
        b_M = np.asarray(b_M, dtype=np.float64).reshape(1, -1)

        pred = X @ self.W_k.T + self.b_k.reshape(1, -1)
        err = pred - Y

        projected = X @ W_M.T + b_M
        score_by_bus = (2.0 / float(self.output_dim)) * err * projected
        if bus_weight is not None:
            bus_weight = np.asarray(bus_weight, dtype=np.float64)
            if bus_weight.shape != score_by_bus.shape:
                raise ValueError(
                    f"bus_weight shape {bus_weight.shape} incompatible with score_by_bus shape {score_by_bus.shape}"
                )
            score_by_bus = score_by_bus * bus_weight
        score = np.sum(score_by_bus, axis=1)

        if normalize_by_n:
            score = score / float(X.shape[0])

        return {
            "score": score,
            "message": self._message(
                "score_contribution",
                score,
                {
                    **self.local_metadata(),
                    "score_min": float(np.min(score)),
                    "score_max": float(np.max(score)),
                    "score_mean": float(np.mean(score)),
                    "score_std": float(np.std(score)),
                },
            ),
        }


def build_clients_from_affine_dataset(
    dataset,
    W_out_in: np.ndarray,
    b: np.ndarray,
    bus_groups: Sequence[np.ndarray],
    feature_mode: str = "precomputed_local_cache",
) -> List[FedClient]:
    X = _to_numpy(dataset.feature).astype(np.float64)
    Y = _to_numpy(dataset.target).astype(np.float64)

    W_out_in = np.asarray(W_out_in, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    output_dim = int(Y.shape[1])
    clients = []

    for cid, buses in enumerate(bus_groups):
        buses = np.asarray(buses, dtype=int)
        clients.append(
            FedClient(
                client_id=cid,
                bus_indices=buses,
                feature_cache=X.copy(),
                target_slice=Y[:, buses].copy(),
                W_k=W_out_in[buses, :].copy(),
                b_k=b[buses].copy(),
                output_dim=output_dim,
                feature_mode=feature_mode,
            )
        )
    return clients


def build_clients_from_raw_dataset(
    raw_dataset,
    target_dataset,
    frozen_backbone,
    W_out_in: np.ndarray,
    b: np.ndarray,
    bus_groups: Sequence[np.ndarray],
    feature_batch_size: int = 128,
    device: str = "cpu",
) -> List[FedClient]:
    """
    Build clients using raw input + frozen backbone.

    H2-2:
        feature_mode = local_frozen_backbone

    The client receives full raw input required by the shared/frozen backbone
    and only its local output target slice.
    """
    raw_X = raw_dataset.feature
    Y = _to_numpy(target_dataset.target).astype(np.float64)

    W_out_in = np.asarray(W_out_in, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    output_dim = int(Y.shape[1])
    clients = []

    for cid, buses in enumerate(bus_groups):
        buses = np.asarray(buses, dtype=int)
        clients.append(
            FedClient(
                client_id=cid,
                bus_indices=buses,
                raw_feature=raw_X,
                frozen_backbone=frozen_backbone,
                target_slice=Y[:, buses].copy(),
                W_k=W_out_in[buses, :].copy(),
                b_k=b[buses].copy(),
                output_dim=output_dim,
                feature_mode="local_frozen_backbone",
                feature_batch_size=feature_batch_size,
                device=device,
            )
        )
    return clients
