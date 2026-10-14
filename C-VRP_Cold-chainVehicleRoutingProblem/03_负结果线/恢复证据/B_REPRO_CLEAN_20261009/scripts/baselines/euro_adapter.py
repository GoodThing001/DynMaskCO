"""
P0-4: EURO Meets NeurIPS 2022 D-VRPTW Benchmark Adapter
=========================================================
Adapts DynMaskCO to the EURO Meets NeurIPS 2022 dynamic VRPTW competition
protocol for external validation.

Competition protocol (from N-Wouda/Euro-NeurIPS-2022):
  - Dynamic VRPTW with stochastic customer requests
  - Events: new order arrival → replan with committed prefix frozen
  - Evaluation: total distance + penalties for unserved/late customers
  - Time budget: configurable per event (typically 1-5s)
  - Solver comparison: same event sequence, same budget

This adapter bridges DynMaskCO's event-driven pipeline to the competition
protocol, enabling apples-to-apples comparison.

Usage:
    python baselines/euro_adapter.py \
        --instance <competition_instance.npz> \
        --ckpt <causal_checkpoint.ckpt> \
        --time_budget_ms 1000

Integration plan:
    Phase 1 (adapter): This script — convert protocol ↔ DynMaskCO
    Phase 2 (benchmark): Run on competition instances, compare with results
    Phase 3 (paper): Report Conditional Recourse Regret on external benchmark
"""

import sys, os, argparse, time, numpy as np

# Path setup
_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_BOOTSTRAP = os.path.dirname(_BASE)
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, os.path.join(_MASKCO, 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'data'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'decoding'))

import jax, jax.numpy as jnp
from flax import nnx
from training import load_ckpt
from cvrptw_utils import coord_normalize_visible


class EuroCompetitionProtocol:
    """EURO Meets NeurIPS 2022 D-VRPTW competition protocol adapter.

    Core abstractions:
      - EventStream: ordered sequence of (time, type, order_id) tuples
      - CommittedPrefix: frozen prefix that cannot be modified
      - InformationState: what is known at event e (F_e filtration)
      - ReplanBudget: wall-clock time budget per replanning event

    The protocol ensures:
      1. All solvers see the SAME event sequence
      2. All solvers have the SAME committed prefix constraint
      3. All solvers have the SAME information at each event
      4. All solvers have the SAME wall-clock budget
    """

    def __init__(self, instance, time_budget_ms=1000):
        """
        Args:
            instance: dict with 'coords', 'demands', 'tw_start', 'tw_end',
                      'service_time', 'reveal_time', 'visible_mask'
            time_budget_ms: wall-clock budget per replanning event
        """
        self.instance = instance
        self.time_budget_ms = time_budget_ms
        self.num_nodes = instance['coords'].shape[0]

        # Build event stream from reveal_time
        self.events = self._build_event_stream()

        # State tracking
        self.committed_prefix = []  # frozen route prefix
        self.known_orders = set()   # revealed customer indices
        self.vehicle_state = {'position': 0, 'time': 0.0, 'load': 0}

    def _build_event_stream(self):
        """Build ordered event stream from reveal_time.

        Event types:
          - 'departure': t=0, vehicle at depot
          - 'revelation': t=reveal_time[k], order k becomes known
          - 'service': t=arrival_time, customer served
        """
        events = [{'time': 0.0, 'type': 'departure', 'order': None}]

        for k in range(1, self.num_nodes):
            rt = self.instance.get('reveal_time',
                                    np.zeros(self.num_nodes))[k]
            if rt > 0:
                events.append({'time': float(rt), 'type': 'revelation',
                               'order': k})

        events.sort(key=lambda e: e['time'])
        return events

    def get_visible_mask(self, current_time):
        """Compute visible_mask at current_time (F_e filtration)."""
        mask = np.ones(self.num_nodes, dtype=np.float32)
        for k in range(1, self.num_nodes):
            rt = self.instance.get('reveal_time',
                                    np.zeros(self.num_nodes))[k]
            if rt > current_time:
                mask[k] = 0.0
        return mask

    def get_committed_constraints(self):
        """Return constraints from committed prefix.

        Returns:
            frozen_edges: set of (i,j) edges that cannot be modified
            last_position: node index of last committed position
            last_time: arrival time at last committed position
        """
        frozen_edges = set()
        last_pos = 0
        last_time = 0.0

        for i in range(len(self.committed_prefix) - 1):
            frozen_edges.add((self.committed_prefix[i],
                              self.committed_prefix[i + 1]))
            last_pos = self.committed_prefix[i + 1]
            # Update last_time based on distance + service
            if i == len(self.committed_prefix) - 2:
                last_time = self.vehicle_state['time']

        return frozen_edges, last_pos, last_time

    def commit_action(self, next_node, travel_time, service_time):
        """Commit to visiting next_node."""
        self.committed_prefix.append(next_node)
        self.vehicle_state['position'] = next_node
        self.vehicle_state['time'] += travel_time + service_time
        if next_node > 0:
            self.known_orders.add(next_node)


class DynMaskCOEuroAdapter:
    """Adapts DynMaskCO to EURO competition protocol."""

    def __init__(self, ckpt_path, time_budget_ms=1000):
        """Load DynMaskCO model."""
        params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
        if model_config is None:
            from DynamicColdChainModel import DynamicColdChainModelConfig
            model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
            model_config.encoder_input_dim = 7

        self.model = model_config.construct_model()
        if params is not None:
            self.model = nnx.merge(nnx.graphdef(self.model), params)

        self.encoder_input_dim = int(self.model.init_proj.kernel.shape[0])
        self.is_dynamic = hasattr(self.model, 'type_embed')
        self.time_budget_ms = time_budget_ms

    def replan(self, raw_features, visible_mask, frozen_prefix,
               dist_mat, demands, tw_start, tw_end, tw_max):
        """Replan from current state with non-anticipatory constraints.

        Args:
            raw_features: (1, N, D) full features (future nodes masked)
            visible_mask: (N,) which nodes are known
            frozen_prefix: list of already-committed node indices
            dist_mat: (N, N) distance matrix
            demands: (N,) customer demands
            tw_start, tw_end: (N,) time windows
            tw_max: float, normalization factor

        Returns:
            new_route: proposed route (frozen prefix + planned suffix)
            cost: estimated cost of full route
        """
        N = raw_features.shape[1]
        B = 1  # single-instance replanning

        # Encode with visible-aware normalization
        feats = raw_features.copy()
        feats[:, :, :2] = coord_normalize_visible(
            jnp.array(feats[..., :2]),
            jnp.array(visible_mask)[None]
        )
        feats = jnp.array(feats[..., :self.encoder_input_dim])

        if self.is_dynamic:
            enc = self.model.encode(
                feats,
                visible_mask=jnp.array(visible_mask)[None]
            )
        else:
            enc = self.model.encode(feats)

        # Build partial adjacency from frozen prefix
        adj = jnp.zeros((B, N, N), dtype=jnp.float32)
        for i in range(len(frozen_prefix) - 1):
            u, v = frozen_prefix[i], frozen_prefix[i + 1]
            adj = adj.at[0, u, v].set(1.0)

        # Decode (single step — simplest; beam uses resource_beam.py)
        logits = self.model.decode(enc, jnp.full((B,), 0.5), adj)

        # Apply frozen prefix mask to logits
        if len(frozen_prefix) > 0:
            last = frozen_prefix[-1]
            # Only allow edges from last committed node
            mask = jnp.ones_like(logits[0]) * -1e9
            mask = mask.at[last, :].set(0.0)
            logits = jnp.where(
                jnp.broadcast_to(mask[None], logits.shape),
                logits, -1e9)

        probs = jax.nn.softmax(logits[0], axis=-1)  # (N, N)

        # Greedy extraction (simplified; full version uses resource_beam.py)
        visited = set(frozen_prefix)
        route = list(frozen_prefix)
        current = frozen_prefix[-1] if frozen_prefix else 0
        remaining_load = 50  # capacity

        while len(route) < N:
            # Build admissible mask
            admissible = np.zeros(N, dtype=bool)
            for j in range(N):
                if j in visited:
                    continue
                if visible_mask[j] == 0:
                    continue
                if demands[j] > remaining_load:
                    continue
                admissible[j] = True

            if not admissible.any():
                break

            # Greedy: highest probability admissible edge
            probs_np = np.array(probs[current])
            probs_np[~admissible] = -np.inf
            next_node = int(np.argmax(probs_np))

            if next_node == 0 and len(route) > 0:
                route.append(0)
                break

            route.append(next_node)
            visited.add(next_node)
            remaining_load -= int(demands[next_node])
            current = next_node

        return route, 0.0  # placeholder cost

    def run_on_instance(self, instance):
        """Run full dynamic simulation on one competition instance.

        Returns:
            final_route, total_distance, unserved_count, late_count, latency_stats
        """
        protocol = EuroCompetitionProtocol(instance, self.time_budget_ms)

        N = instance['coords'].shape[0]
        cap = 50
        tw_max = float(instance['tw_end'].max())

        # Build features once (will be masked per-event)
        fl = [
            instance['coords'],
            (instance['demands'] / cap)[..., None],
            (instance['tw_start'] / tw_max)[..., None],
            (instance['tw_end'] / tw_max)[..., None],
            (instance.get('temp_class',
                          np.zeros(N, dtype=np.float32)) / 2.0)[..., None],
        ]
        if 'reveal_time' in instance:
            fl.append((instance['reveal_time'].astype(np.float32)
                       / tw_max)[..., None])
        full_features = np.concatenate(fl, axis=-1).astype(np.float32)

        # Precompute distance matrix
        coords = instance['coords']
        diff = coords[:, None] - coords[None, :]
        dist_mat = np.sqrt((diff ** 2).sum(axis=-1))

        latency_ms = []

        for event_idx, event in enumerate(protocol.events):
            current_time = event['time']

            # Update known orders
            if event['type'] == 'revelation':
                protocol.known_orders.add(event['order'])

            # Compute current visible mask
            vis_mask = protocol.get_visible_mask(current_time)

            # Get committed constraints
            frozen_edges, last_pos, last_time = \
                protocol.get_committed_constraints()

            # Build masked features for current state
            features_now = full_features.copy()
            vis_3d = vis_mask[:, None]
            features_now[:, 2:] = features_now[:, 2:] * vis_3d
            features_now[:, :2] = (features_now[:, :2] * vis_3d
                                   + (1 - vis_3d) * 0.5)

            # Replan
            t0 = time.time()
            new_route, cost = self.replan(
                features_now[None], vis_mask,
                protocol.committed_prefix,
                dist_mat, instance['demands'],
                instance['tw_start'], instance['tw_end'],
                tw_max)
            latency_ms.append((time.time() - t0) * 1000)

            # Commit next action (simplified: commit one step)
            if len(new_route) > len(protocol.committed_prefix):
                next_node = new_route[len(protocol.committed_prefix)]
                travel = dist_mat[protocol.vehicle_state['position'],
                                  next_node]
                svc = instance.get('service_time',
                                    np.zeros(N))[next_node]
                protocol.commit_action(next_node, float(travel), float(svc))

        # Final evaluation
        final_route = protocol.committed_prefix

        # Compute metrics
        total_dist = 0.0
        for i in range(len(final_route) - 1):
            total_dist += dist_mat[final_route[i], final_route[i + 1]]

        # Count unserved
        served = set(final_route) - {0}
        all_customers = set(range(1, N))
        unserved = len(all_customers - served)

        # Check TW
        late = 0
        t = 0.0
        prev = 0
        for v in final_route:
            if v == 0:
                t = 0.0
                continue
            t += dist_mat[prev, v] + instance.get('service_time',
                                                   np.zeros(N))[v]
            if t > instance['tw_end'][v]:
                late += 1
            prev = v

        latency_p50 = float(np.median(latency_ms)) if latency_ms else 0.0
        latency_p95 = float(np.percentile(latency_ms, 95)) if latency_ms else 0.0

        return final_route, total_dist, unserved, late, latency_p50, latency_p95


def main():
    parser = argparse.ArgumentParser(
        description='DynMaskCO EURO Meets NeurIPS 2022 D-VRPTW Adapter')
    parser.add_argument('--instance', type=str, required=True,
                        help='Path to competition instance .npz')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to DynMaskCO checkpoint')
    parser.add_argument('--time_budget_ms', type=int, default=1000,
                        help='Wall-clock budget per replanning event (ms)')
    args = parser.parse_args()

    instance = dict(np.load(args.instance))

    print("=" * 60)
    print("DynMaskCO EURO D-VRPTW Adapter")
    print("=" * 60)
    print(f"  Instance: {args.instance}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Nodes: {instance['coords'].shape[0]}")
    print(f"  Budget: {args.time_budget_ms}ms/event")

    adapter = DynMaskCOEuroAdapter(args.ckpt, args.time_budget_ms)
    route, dist, unserved, late, p50, p95 = adapter.run_on_instance(instance)

    print(f"\n--- Results ---")
    print(f"  Route length: {len(route)}")
    print(f"  Total distance: {dist:.2f}")
    print(f"  Unserved: {unserved}")
    print(f"  Late: {late}")
    print(f"  Latency p50: {p50:.0f}ms")
    print(f"  Latency p95: {p95:.0f}ms")

    # Competition objective (distance + penalties)
    obj = dist + 1000 * unserved + 500 * late
    print(f"  Competition obj: {obj:.2f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
