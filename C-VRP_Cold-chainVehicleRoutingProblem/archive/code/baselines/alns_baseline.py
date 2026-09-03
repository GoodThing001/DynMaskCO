"""
纯 Python ALNS baseline for DCCVRP (无外部依赖)。

用法:
    python baselines/alns_baseline.py \
        --data dcc_50_r1_edod05_test.npz --capacity 50 \
        --iterations 5000 --num_instances 16
"""

import sys, os, argparse, time, numpy as np
_BASE = os.path.dirname(os.path.abspath(__file__))
_CVRPTW = os.path.dirname(os.path.dirname(_BASE))

class ALNS:
    """Adaptive Large Neighborhood Search for CVRPTW."""
    def __init__(self, coords, demands, tw_start, tw_end, service_time, capacity, speed=1.0):
        self.coords = coords
        self.demands = demands
        self.tw_start = tw_start
        self.tw_end = tw_end
        self.service_time = service_time
        self.capacity = capacity
        self.speed = speed
        self.nodes = len(coords)
        diff = coords[:,None,:] - coords[None,:,:]
        self.dist = np.sqrt((diff**2).sum(axis=-1))

    def _feasible_route(self, route):
        """检查路径 TW+容量可行性。"""
        t, load, prev = 0, 0, 0
        for node in route:
            if node == 0: t, load, prev = 0, 0, 0; continue
            t += self.service_time[prev] + self.dist[prev,node]/self.speed
            t = max(t, self.tw_start[node])
            if t > self.tw_end[node] + 1e-6: return False
            load += self.demands[node]
            if load > self.capacity: return False
            prev = node
        return True

    def _route_cost(self, route):
        c, prev = 0, 0
        for n in route:
            c += self.dist[prev,n]; prev = n
        return c

    def _cost(self, solution):
        return sum(self._route_cost(r) for r in solution)

    def _greedy_init(self):
        unvisited = set(range(1, self.nodes))
        solution = []
        while unvisited:
            route = [0]; t, load, cur = 0, 0, 0
            while unvisited:
                best, best_d = None, float('inf')
                for j in unvisited:
                    if load + self.demands[j] > self.capacity: continue
                    arr = t + self.service_time[cur] + self.dist[cur,j]/self.speed
                    arr = max(arr, self.tw_start[j])
                    if arr <= self.tw_end[j] and self.dist[cur,j] < best_d:
                        best_d = self.dist[cur,j]; best = j
                if best is None: break
                route.append(best); unvisited.remove(best)
                t = max(t + self.service_time[cur] + self.dist[cur,best]/self.speed, self.tw_start[best])
                load += self.demands[best]; cur = best
            route.append(0)
            if len(route) > 2: solution.append(route)
        return solution

    def _random_removal(self, solution, q):
        """随机删除 q 个客户。"""
        all_nodes = [(ri,pi) for ri,r in enumerate(solution) for pi,n in enumerate(r) if n>0]
        if len(all_nodes) <= q: return solution, []
        removed = [all_nodes[i] for i in np.random.choice(len(all_nodes), q, replace=False)]
        new_sol = [[n for n in r] for r in solution]
        removed_nodes = []
        for ri, pi in sorted(removed, key=lambda x: -x[1]):  # reverse order
            removed_nodes.append(new_sol[ri].pop(pi))
        new_sol = [r for r in new_sol if len(r) > 2]  # 移除空路径
        return new_sol, removed_nodes

    def _greedy_insert(self, solution, nodes):
        """贪心插入节点。"""
        for node in nodes:
            best_cost, best_pos = float('inf'), None
            for ri in range(len(solution)):
                for pi in range(1, len(solution[ri])):
                    cand = solution[ri][:pi] + [node] + solution[ri][pi:]
                    if self._feasible_route(cand):
                        c = self._route_cost(cand)
                        if c < best_cost:
                            best_cost, best_pos = c, (ri, pi)
            if best_pos:
                ri, pi = best_pos
                solution[ri] = solution[ri][:pi] + [node] + solution[ri][pi:]
            else:
                solution.append([0, node, 0])
        return solution

    def solve(self, iterations=5000):
        np.random.seed(42)
        current = self._greedy_init()
        best = [r[:] for r in current]
        best_cost = self._cost(best)

        for it in range(iterations):
            q = max(1, int(sum(len(r)-2 for r in current) * np.random.uniform(0.1, 0.4)))
            destroyed, removed = self._random_removal(current, q)
            repaired = self._greedy_insert(destroyed, removed)
            new_cost = self._cost(repaired)

            # 模拟退火接受
            T = 10.0 * (1 - it/iterations)
            if new_cost < self._cost(current) or np.random.random() < np.exp(-(new_cost - self._cost(current))/max(T,1e-6)):
                current = repaired
                if new_cost < best_cost:
                    best = [r[:] for r in repaired]
                    best_cost = new_cost
        return best, best_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=5000)
    parser.add_argument('--num_instances', type=int, default=16)
    args = parser.parse_args()

    dataset = dict(np.load(args.data))
    N = min(args.num_instances, dataset['coords'].shape[0])

    print(f"ALNS Baseline: {N} instances, {args.iterations} iters")
    costs, times = [], []
    for i in range(N):
        t0 = time.time()
        alns = ALNS(dataset['coords'][i], dataset['demands'][i],
                    dataset['tw_start'][i], dataset['tw_end'][i],
                    dataset.get('service_time', np.zeros_like(dataset['demands']))[i],
                    args.capacity)
        sol, cost = alns.solve(args.iterations)
        elapsed = time.time() - t0
        costs.append(cost)
        times.append(elapsed)
        if (i+1) % 4 == 0:
            print(f"  {i+1}/{N} | cost={np.mean(costs):.2f} | {np.mean(times):.1f}s/inst")

    print(f"\n--- ALNS Results ---")
    print(f"  Avg cost: {np.mean(costs):.2f}")
    print(f"  Avg time:  {np.mean(times):.1f}s/inst")
    print(f"  Ref cost: {dataset['opt_costs'][:N].mean():.2f}")
    print(f"  MaskCO (best): 14.67 (feas 93.8%)")


if __name__ == '__main__':
    main()
