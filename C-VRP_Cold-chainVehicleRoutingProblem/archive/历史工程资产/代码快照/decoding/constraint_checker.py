"""
Constraint Checker for Beam Search with Constraint Cutting

This module implements feasibility checking for partial beam search states.
Used to prune infeasible branches early in beam search (Week 1, v6).

References:
- Shati et al. "Beam Search with Constraint Cutting" (SoCS 2024)
- Week 1 implementation plan in v6/执行方案_v6.md
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import time


class ConstraintChecker:
    """
    Fast heuristic-based constraint checking for beam search states.

    Checks if a partial route state can be extended to a complete feasible solution.
    Three-level checking:
    1. Capacity feasibility (remaining capacity >= remaining demand)
    2. Time window feasibility (greedy EDD ordering)
    3. Return-to-depot feasibility (can return within depot TW)
    """

    def __init__(
        self,
        coords: np.ndarray,        # (N, 2) node coordinates
        demands: np.ndarray,       # (N,) demand at each node
        tw_start: np.ndarray,      # (N,) time window start
        tw_end: np.ndarray,        # (N,) time window end
        service_time: np.ndarray,  # (N,) service time at each node
        capacity: float,           # vehicle capacity
        speed: float = 1.0,        # travel speed (distance/time)
        time_margin: float = 0.0,  # safety margin for TW (default 0, can set 0.05)
        verbose: bool = False
    ):
        self.coords = coords
        self.demands = demands
        self.tw_start = tw_start
        self.tw_end = tw_end
        self.service_time = service_time
        self.capacity = capacity
        self.speed = speed
        self.time_margin = time_margin
        self.verbose = verbose

        # Precompute distance matrix
        self.num_nodes = len(coords)
        self.dist_mat = self._compute_distance_matrix(coords)

        # Statistics
        self.stats = {
            'total_checks': 0,
            'capacity_failures': 0,
            'tw_failures': 0,
            'depot_return_failures': 0,
            'successes': 0,
            'total_time': 0.0
        }

    def _compute_distance_matrix(self, coords: np.ndarray) -> np.ndarray:
        """Compute Euclidean distance matrix."""
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        return dist

    def is_extendable_to_feasible(
        self,
        visited: np.ndarray,       # (N,) boolean mask of visited nodes
        current_node: int,         # current position
        current_time: float,       # current time
        remaining_capacity: float, # remaining vehicle capacity
        check_mode: str = 'fast'   # 'fast' or 'thorough'
    ) -> Tuple[bool, str]:
        """
        Check if the current partial state can be extended to a feasible complete solution.

        Args:
            visited: Boolean mask indicating which nodes have been visited
            current_node: Current node ID
            current_time: Current time
            remaining_capacity: Remaining capacity
            check_mode: 'fast' (heuristic) or 'thorough' (not implemented yet)

        Returns:
            (is_feasible, reason): Boolean feasibility and reason string
        """
        start_time = time.time()
        self.stats['total_checks'] += 1

        # Get unvisited nodes (excluding depot=0 from middle of route)
        unvisited = np.where(~visited)[0]
        unvisited = unvisited[unvisited > 0]  # Exclude depot from unvisited

        if len(unvisited) == 0:
            # All nodes visited, just check return to depot
            travel_time = self.dist_mat[current_node, 0] / self.speed
            arrival_depot = current_time + travel_time
            feasible = arrival_depot <= self.tw_end[0]
            reason = 'all_visited_ok' if feasible else 'cannot_return_to_depot'
            if not feasible:
                self.stats['depot_return_failures'] += 1
            else:
                self.stats['successes'] += 1
            self.stats['total_time'] += time.time() - start_time
            return feasible, reason

        # === Level 1: Capacity Check ===
        remaining_demand = np.sum(self.demands[unvisited])
        if remaining_capacity < remaining_demand:
            self.stats['capacity_failures'] += 1
            self.stats['total_time'] += time.time() - start_time
            return False, 'insufficient_capacity'

        # === Level 2: Time Window Greedy Check ===
        # Sort unvisited by TW end time (EDD - Earliest Due Date)
        edd_order = unvisited[np.argsort(self.tw_end[unvisited])]

        sim_time = current_time
        sim_node = current_node

        for next_node in edd_order:
            # Travel to next node
            travel_time = self.dist_mat[sim_node, next_node] / self.speed
            arrival = sim_time + travel_time

            # Check TW feasibility (with margin)
            if arrival > self.tw_end[next_node] - self.time_margin:
                self.stats['tw_failures'] += 1
                self.stats['total_time'] += time.time() - start_time
                return False, f'tw_violation_at_node_{next_node}'

            # Wait if early
            service_start = max(arrival, self.tw_start[next_node])
            sim_time = service_start + self.service_time[next_node]
            sim_node = next_node

        # === Level 3: Return to Depot Check ===
        travel_to_depot = self.dist_mat[sim_node, 0] / self.speed
        arrival_depot = sim_time + travel_to_depot

        if arrival_depot > self.tw_end[0] - self.time_margin:
            self.stats['depot_return_failures'] += 1
            self.stats['total_time'] += time.time() - start_time
            return False, 'cannot_return_to_depot_in_time'

        # All checks passed
        self.stats['successes'] += 1
        self.stats['total_time'] += time.time() - start_time
        return True, 'feasible'

    def check_single_extension(
        self,
        visited: np.ndarray,
        current_node: int,
        current_time: float,
        remaining_capacity: float,
        candidate_node: int
    ) -> Tuple[bool, str]:
        """
        Check if adding candidate_node to current state is immediately feasible.
        (Lighter check than full extendability)

        Returns:
            (is_feasible, reason)
        """
        # Capacity check
        if remaining_capacity < self.demands[candidate_node]:
            return False, 'insufficient_capacity'

        # Direct TW check
        travel_time = self.dist_mat[current_node, candidate_node] / self.speed
        arrival = current_time + travel_time

        if arrival > self.tw_end[candidate_node] - self.time_margin:
            return False, f'tw_violation'

        return True, 'ok'

    def get_statistics(self) -> Dict[str, Any]:
        """Return checking statistics."""
        total = self.stats['total_checks']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'success_rate': self.stats['successes'] / total,
            'capacity_fail_rate': self.stats['capacity_failures'] / total,
            'tw_fail_rate': self.stats['tw_failures'] / total,
            'depot_fail_rate': self.stats['depot_return_failures'] / total,
            'avg_time_ms': self.stats['total_time'] / total * 1000
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        for key in self.stats:
            if isinstance(self.stats[key], (int, float)):
                self.stats[key] = 0


# === Optional: OR-Tools CP-SAT Integration (Day 4) ===
class ORToolsConstraintChecker:
    """
    OR-Tools CP-SAT based constraint checker.
    More precise but slower than heuristic checker.

    Usage: Only for top-K candidates when fast heuristic is insufficient.
    """

    def __init__(self, *args, **kwargs):
        # Placeholder for OR-Tools integration
        raise NotImplementedError(
            "OR-Tools CP-SAT integration is optional (Day 4). "
            "Use ConstraintChecker (fast heuristic) for now."
        )

    def is_extendable_to_feasible(self, *args, **kwargs):
        """CP-SAT based feasibility check."""
        # TODO: Implement using ortools.sat.python.cp_model
        pass


# === Testing / Debug Utilities ===
def test_constraint_checker():
    """Unit test for constraint checker."""
    print("Testing ConstraintChecker...")

    # Simple 5-node instance
    coords = np.array([
        [0.5, 0.5],  # depot
        [0.2, 0.3],  # node 1
        [0.8, 0.7],  # node 2
        [0.4, 0.9],  # node 3
        [0.6, 0.1],  # node 4
    ])
    demands = np.array([0, 5, 10, 8, 7])
    tw_start = np.array([0, 1, 2, 3, 4])
    tw_end = np.array([100, 10, 15, 20, 25])
    service_time = np.array([0, 0.5, 0.5, 0.5, 0.5])
    capacity = 30.0

    checker = ConstraintChecker(
        coords, demands, tw_start, tw_end, service_time, capacity,
        verbose=True
    )

    # Test 1: Empty route (all unvisited)
    visited = np.array([True, False, False, False, False])  # Only depot visited
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=0, current_time=0.0, remaining_capacity=30.0
    )
    print(f"Test 1 (empty route): feasible={feasible}, reason={reason}")

    # Test 2: Partial route with tight capacity
    visited = np.array([True, True, True, False, False])  # 1,2 visited
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=2, current_time=5.0, remaining_capacity=15.0
    )
    print(f"Test 2 (partial, tight capacity): feasible={feasible}, reason={reason}")

    # Test 3: Insufficient capacity
    visited = np.array([True, True, False, False, False])
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=1, current_time=3.0, remaining_capacity=10.0  # < 10+8+7
    )
    print(f"Test 3 (insufficient capacity): feasible={feasible}, reason={reason}")
    assert not feasible and 'capacity' in reason

    # Statistics
    print("\nStatistics:")
    for k, v in checker.get_statistics().items():
        print(f"  {k}: {v}")

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_constraint_checker()
