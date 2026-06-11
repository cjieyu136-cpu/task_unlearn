"""
utils/fed_secure_aggregation.py

Stage 3H H3-1 secure aggregation mock utilities.

This is NOT cryptographic secure aggregation.

Purpose
-------
Provide an API-level mock that hides per-client vectors from saved server-side
summaries and exposes only aggregated sums.

Supported mode:
    none
        no secure aggregation mock.

    mock_sum
        server receives only aggregate sums in the public output tables.
        In this single-machine simulation, client contributions are still
        computed in memory, but per-client raw score/gradient/M vectors are not
        saved to runtime payload summaries.

What this does:
    - sums client contribution vectors
    - records aggregate payload sizes
    - provides redacted per-client metadata only

What this does not do:
    - encryption
    - masking protocol
    - threshold secure aggregation
    - protection against malicious server in real deployment
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd


@dataclass
class SecureAggRecord:
    aggregate_name: str
    num_clients: int
    vector_shape: str
    total_elements: int
    metadata: Dict[str, Any]


class MockSecureSumAggregator:
    """
    Mock secure-sum aggregator.

    Client vectors are added to an aggregate. The public record exposes only:
        aggregate_name
        num_clients
        vector shape
        total transmitted elements
        metadata

    It does not expose per-client vector values.
    """

    def __init__(self, mode: str = "none"):
        self.mode = str(mode)
        self._sums: Dict[str, np.ndarray] = {}
        self._counts: Dict[str, int] = {}
        self._elements: Dict[str, int] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add(self, name: str, client_id: int, vector, metadata: Optional[Dict[str, Any]] = None):
        arr = np.asarray(vector, dtype=np.float64)
        key = str(name)

        if key not in self._sums:
            self._sums[key] = np.zeros_like(arr, dtype=np.float64)
            self._counts[key] = 0
            self._elements[key] = 0
            self._metadata[key] = dict(metadata or {})
        else:
            if self._sums[key].shape != arr.shape:
                raise ValueError(
                    f"Shape mismatch for aggregate {key}: "
                    f"expected {self._sums[key].shape}, got {arr.shape}"
                )

        self._sums[key] += arr
        self._counts[key] += 1
        self._elements[key] += int(arr.size)

    def sum(self, name: str) -> np.ndarray:
        return self._sums[str(name)]

    def public_records(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for key, arr in self._sums.items():
            rows.append({
                "message_type": f"secure_agg_{key}",
                "secure_agg_mode": self.mode,
                "num_clients": int(self._counts[key]),
                "payload_shape": str(tuple(arr.shape)),
                "num_elements": int(self._elements[key]),
                **{f"meta_{k}": v for k, v in self._metadata[key].items()},
            })
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Any]:
        out = {"secure_agg_mode": self.mode}
        for key, arr in self._sums.items():
            out[f"{key}_num_clients"] = int(self._counts[key])
            out[f"{key}_aggregate_shape"] = str(tuple(arr.shape))
            out[f"{key}_total_elements"] = int(self._elements[key])
        return out


def redact_client_table(df: pd.DataFrame, secure_agg_mode: str, keep_columns=None) -> pd.DataFrame:
    """
    Redact a per-client table for mock secure aggregation mode.

    This keeps only metadata columns by default and removes value/stat columns
    that may reveal contribution information.
    """
    if str(secure_agg_mode) != "mock_sum":
        return df

    if keep_columns is None:
        keep_columns = [
            "client_id",
            "bus_indices",
            "num_buses",
            "num_samples",
            "feature_dim",
            "theta_k_elements",
            "feature_mode",
        ]

    cols = [c for c in keep_columns if c in df.columns]
    out = df[cols].copy()
    out["secure_agg_mode"] = str(secure_agg_mode)
    out["redacted"] = True
    return out
