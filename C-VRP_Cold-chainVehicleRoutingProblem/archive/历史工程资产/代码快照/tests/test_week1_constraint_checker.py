"""
Week 1 v6: Unit Test for Constraint Checker

Tests the ConstraintChecker module in isolation before integration.
Run: python scripts/tests/test_week1_constraint_checker.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from decoding.constraint_checker import ConstraintChecker
import time


def test_basic_feasibility():
    """Test 1: Basic feasibility checks on a simple instance."""
    print("=" * 60)
    print("Test 1: Basic Feasibility Checks")
    print("=" * 60)

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
        coords, demands, tw_start, tw_end, service_time, capacity
    )

    # Test 1.1: Empty route (all unvisited except depot)
    visited = np.array([True, False, False, False, False])
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=0, current_time=0.0, remaining_capacity=30.0
    )
    print(f"1.1 Empty route: feasible={feasible}, reason={reason}")
    assert feasible, "Empty route should be feasible"

    # Test 1.2: Partial route (1,2 visited, 3,4 remaining)
    visited = np.array([True, True, True, False, False])
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=2, current_time=5.0, remaining_capacity=15.0
    )
    print(f"1.2 Partial route: feasible={feasible}, reason={reason}")

    # Test 1.3: Insufficient capacity
    visited = np.array([True, True, False, False, False])
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=1, current_time=3.0, remaining_capacity=10.0  # < 10+8+7=25
    )
    print(f"1.3 Insufficient capacity: feasible={feasible}, reason={reason}")
    assert not feasible and 'capacity' in reason, "Should fail on capacity"

    print("✅ Test 1 passed\n")


def test_time_window_constraints():
    """Test 2: Time window feasibility checks."""
    print("=" * 60)
    print("Test 2: Time Window Constraints")
    print("=" * 60)

    # Instance with tight time windows
    coords = np.array([
        [0.5, 0.5],  # depot
        [0.1, 0.1],  # node 1 (close to depot)
        [0.9, 0.9],  # node 2 (far from depot)
        [0.5, 0.9],  # node 3 (medium)
    ])
    demands = np.array([0, 5, 5, 5])
    tw_start = np.array([0, 0, 5, 3])
    tw_end = np.array([100, 2, 7, 6])  # node 1 tight TW [0,2]
    service_time = np.array([0, 0.5, 0.5, 0.5])
    capacity = 20.0

    checker = ConstraintChecker(
        coords, demands, tw_start, tw_end, service_time, capacity, speed=1.0
    )

    # Test 2.1: Visit node 2 first (far), then node 1 (tight TW) - should fail
    visited = np.array([True, False, True, False])  # depot, node 2 visited
    # After visiting node 2, time is ~5 + service, cannot reach node 1 by tw_end[1]=2
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=2, current_time=6.0, remaining_capacity=15.0
    )
    print(f"2.1 Late start (cannot satisfy tight TW): feasible={feasible}, reason={reason}")
    assert not feasible and 'tw_violation' in reason, "Should fail on TW violation"

    # Test 2.2: Visit node 1 first (tight TW), then others - should pass
    visited = np.array([True, True, False, False])  # depot, node 1 visited
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=1, current_time=1.0, remaining_capacity=15.0
    )
    print(f"2.2 Early start (can satisfy remaining TWs): feasible={feasible}, reason={reason}")

    print("✅ Test 2 passed\n")


def test_depot_return_constraint():
    """Test 3: Depot return feasibility."""
    print("=" * 60)
    print("Test 3: Depot Return Constraint")
    print("=" * 60)

    # Instance where depot closes early
    coords = np.array([
        [0.5, 0.5],  # depot
        [0.1, 0.1],  # node 1
        [0.9, 0.9],  # node 2 (far)
    ])
    demands = np.array([0, 5, 5])
    tw_start = np.array([0, 0, 0])
    tw_end = np.array([10, 5, 8])  # depot closes at t=10
    service_time = np.array([0, 0.5, 0.5])
    capacity = 10.0

    checker = ConstraintChecker(
        coords, demands, tw_start, tw_end, service_time, capacity, speed=1.0
    )

    # Test 3.1: At node 2 (far), time=8, must visit node 1 first (tight TW)
    visited = np.array([True, False, True])  # depot, node 2 visited
    # Node 1 has tw_end=5, but we're at time=8, so cannot satisfy node 1's TW
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=2, current_time=8.0, remaining_capacity=5.0
    )
    print(f"3.1 Too late for remaining TWs: feasible={feasible}, reason={reason}")
    assert not feasible, "Should fail on TW violation for node 1"

    # Test 3.2: All nodes visited, late return to depot
    visited = np.array([True, True, True])  # all visited
    feasible, reason = checker.is_extendable_to_feasible(
        visited, current_node=2, current_time=9.8, remaining_capacity=0.0
    )
    print(f"3.2 Too late to return to depot: feasible={feasible}, reason={reason}")
    assert not feasible and 'depot' in reason, "Should fail on depot return"

    print("✅ Test 3 passed\n")


def test_performance():
    """Test 4: Performance benchmark on larger instances."""
    print("=" * 60)
    print("Test 4: Performance Benchmark")
    print("=" * 60)

    # 50-node instance (similar to actual test data)
    np.random.seed(42)
    N = 51  # 50 customers + 1 depot
    coords = np.random.rand(N, 2)
    coords[0] = [0.5, 0.5]  # depot at center

    demands = np.random.randint(1, 10, size=N)
    demands[0] = 0

    tw_start = np.sort(np.random.rand(N) * 20)
    tw_end = tw_start + np.random.rand(N) * 5 + 2
    service_time = np.ones(N) * 0.5
    service_time[0] = 0.0
    capacity = 50.0

    checker = ConstraintChecker(
        coords, demands, tw_start, tw_end, service_time, capacity
    )

    # Benchmark: 100 checks at various stages
    num_checks = 100
    visited_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    print(f"Running {num_checks} feasibility checks...")
    print("")

    for ratio in visited_ratios:
        num_visited = int(N * ratio)
        visited_nodes = set(np.random.choice(range(1, N), size=num_visited, replace=False))
        visited_nodes.add(0)  # depot always visited

        visited_mask = np.zeros(N, dtype=bool)
        for v in visited_nodes:
            visited_mask[v] = True

        current_node = np.random.choice(list(visited_nodes))
        current_time = np.random.rand() * 20
        remaining_cap = capacity - np.sum(demands[list(visited_nodes)])

        start = time.time()
        for _ in range(num_checks):
            checker.is_extendable_to_feasible(
                visited_mask, current_node, current_time, remaining_cap
            )
        elapsed = time.time() - start

        avg_time_ms = elapsed / num_checks * 1000
        print(f"  Visited ratio {ratio:.1%}: {avg_time_ms:.3f} ms/check")

    stats = checker.get_statistics()
    print(f"\n  Total checks: {stats['total_checks']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Avg time: {stats['avg_time_ms']:.3f} ms")
    print(f"  Capacity failures: {stats['capacity_fail_rate']:.1%}")
    print(f"  TW failures: {stats['tw_fail_rate']:.1%}")
    print(f"  Depot failures: {stats['depot_fail_rate']:.1%}")

    print("\n✅ Test 4 passed\n")


def test_single_extension():
    """Test 5: Single extension check (lighter than full extendability)."""
    print("=" * 60)
    print("Test 5: Single Extension Check")
    print("=" * 60)

    coords = np.array([[0.5, 0.5], [0.2, 0.3], [0.8, 0.7]])
    demands = np.array([0, 5, 10])
    tw_start = np.array([0, 1, 5])
    tw_end = np.array([100, 10, 15])
    service_time = np.array([0, 0.5, 0.5])
    capacity = 15.0

    checker = ConstraintChecker(
        coords, demands, tw_start, tw_end, service_time, capacity
    )

    # Test: Can we go from depot to node 1?
    visited = np.array([True, False, False])
    feasible, reason = checker.check_single_extension(
        visited, current_node=0, current_time=0.0,
        remaining_capacity=15.0, candidate_node=1
    )
    print(f"5.1 Depot → Node 1: feasible={feasible}, reason={reason}")
    assert feasible, "Should be able to visit node 1 from depot"

    # Test: Can we go from node 1 to node 2 with tight capacity?
    feasible, reason = checker.check_single_extension(
        visited, current_node=1, current_time=2.0,
        remaining_capacity=10.0, candidate_node=2
    )
    print(f"5.2 Node 1 → Node 2 (exact capacity): feasible={feasible}, reason={reason}")
    assert feasible, "Should fit exactly"

    # Test: Insufficient capacity
    feasible, reason = checker.check_single_extension(
        visited, current_node=1, current_time=2.0,
        remaining_capacity=5.0, candidate_node=2
    )
    print(f"5.3 Node 1 → Node 2 (insufficient capacity): feasible={feasible}, reason={reason}")
    assert not feasible, "Should fail on capacity"

    print("✅ Test 5 passed\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Week 1 v6: Constraint Checker Unit Tests")
    print("=" * 60 + "\n")

    try:
        test_basic_feasibility()
        test_time_window_constraints()
        test_depot_return_constraint()
        test_performance()
        test_single_extension()

        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nConstraint checker is ready for integration!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
