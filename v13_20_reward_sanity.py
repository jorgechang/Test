#!/usr/bin/env python3
"""Pure-Python sanity checks for the exact v13.20 dense state reward."""
from __future__ import annotations

import math

PROX_SCALE = 0.5
SIGMA = 2.0
IN_TOL_REWARD = 0.5
GAMMA = 332.0 / 333.0


def reward(error_m: float, in_tolerance: bool = False, true_terminal: bool = False) -> float:
    if true_terminal:
        return 0.0
    return PROX_SCALE * math.exp(-error_m / SIGMA) + (
        IN_TOL_REWARD if in_tolerance else 0.0
    )


def main() -> int:
    errors = [0.0, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0]
    values = [reward(e) for e in errors]
    print("[STATE REWARD] error -> reward")
    for e, r in zip(errors, values):
        print(f"  E={e:4.2f} m -> r={r:.6f}")

    assert all(a > b for a, b in zip(values, values[1:])), values
    assert 0.0 < values[-1] < values[0] <= PROX_SCALE + 1e-12
    assert abs(reward(0.0, True) - 1.0) < 1e-12
    assert reward(0.10, True) > reward(0.10, False)
    assert reward(0.0, True, true_terminal=True) == 0.0

    # A persistent improvement creates a persistent value difference; unlike PBRS,
    # this does not telescope away when the critic fits the reward exactly.
    delta_r = reward(1.0) - reward(2.0)
    asymptotic_delta_v = delta_r / (1.0 - GAMMA)
    print(f"[STATE REWARD] r(E=1)-r(E=2) = {delta_r:.6f}")
    print(f"[STATE REWARD] persistent discounted value gap ~= {asymptotic_delta_v:.3f}")
    assert delta_r > 0.0 and asymptotic_delta_v > 10.0

    # Better state should dominate worse state over a finite 500-step horizon too.
    horizon = 500
    geom = sum(GAMMA**t for t in range(horizon))
    finite_gap = delta_r * geom
    print(f"[STATE REWARD] 500-step persistent value gap ~= {finite_gap:.3f}")
    assert finite_gap > 1.0

    print("[STATE REWARD] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
