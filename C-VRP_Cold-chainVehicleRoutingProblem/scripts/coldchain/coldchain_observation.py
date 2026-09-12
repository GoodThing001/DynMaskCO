"""Causal cold-chain order observations for strict-online decision inputs."""

from __future__ import annotations

import numpy as np


def build_visible_coldchain_observation(
    temp_class,
    initial_quality,
    reveal_time,
    clock,
    *,
    depot_id=0,
):
    """Mask unrevealed order attributes while preserving shape and depot identity.

    Future temperature class uses the explicit sentinel ``-1`` and future
    initial quality uses ``0``.  The true arrays remain owned by the environment
    for transition/evaluation only and are never returned by this function.
    """

    temp_class = np.asarray(temp_class, dtype=np.int32)
    initial_quality = np.asarray(initial_quality, dtype=np.float32)
    reveal_time = np.asarray(reveal_time, dtype=np.float32)
    if not (temp_class.shape == initial_quality.shape == reveal_time.shape):
        raise ValueError("cold-chain order arrays must have identical shapes")
    if not 0 <= int(depot_id) < temp_class.size:
        raise ValueError("depot_id is out of range")
    visible = reveal_time <= float(clock) + 1e-6
    visible = visible.copy()
    visible[int(depot_id)] = True
    masked_temp = np.where(visible, temp_class, -1).astype(np.int32, copy=False)
    masked_quality = np.where(visible, initial_quality, 0.0).astype(np.float32, copy=False)
    return {
        'visible_mask': visible,
        'temp_class': masked_temp,
        'initial_quality': masked_quality,
    }
