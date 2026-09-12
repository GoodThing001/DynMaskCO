"""在 reveal 事件捕获 fleet 状态，解释为什么 late-reveal tight-window 客户无法被服务。"""
import argparse, os, sys
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import JF1HRepairReplanner


class DiagReplanner(JF1HRepairReplanner):
    def __init__(self, watch_customers, dataset, inst_idx):
        super().__init__()
        self.watch = set(watch_customers)
        self.dataset = dataset
        self.inst = inst_idx

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        # 先记录当前 reveal 的 fleet 状态（若存在 watch 客户刚 reveal 且未服务）
        newly_revealed = [c for c in self.watch
                          if env.reveal_time[inst_idx, c] <= clock + 1e-6
                          and not served_mask[int(c)]]
        if newly_revealed:
            statuses = {}
            for v in vehicles:
                s = v.status
                statuses[s] = statuses.get(s, 0) + 1
            print(f"\n[clock={clock:.2f}] watch={newly_revealed} fleet={statuses}")
            dmat = env.dist_mat[inst_idx]
            for v in vehicles:
                if v.status == 'committed':
                    node = v.committed_next
                    finish = v.committed_finish
                    print(f"  v{v.vehicle_id} committed->{node} finish={finish:.2f} "
                          f"load={v.current_load:.0f}")
                elif v.status == 'ready':
                    print(f"  v{v.vehicle_id} ready@node{v.current_node} ready_time={v.ready_time:.2f}")
                elif v.status == 'idle':
                    print(f"  v{v.vehicle_id} idle@node{v.current_node}")
            for c in newly_revealed:
                tw_e = env.tw_end[inst_idx, c]
                d_depot = dmat[0, c]
                # 各车（含 committed 完成后）到该客户的最早到达
                best_arrive = float('inf')
                for v in vehicles:
                    if v.status in ('idle',):
                        node, t = v.current_node, clock
                    elif v.status in ('ready',):
                        node, t = v.current_node, v.ready_time
                    elif v.status == 'committed':
                        node, t = v.committed_next, v.committed_finish
                    else:
                        continue
                    arrive = t + dmat[node, c]
                    if arrive < best_arrive:
                        best_arrive = arrive
                print(f"  cust {c}: tw_end={tw_e:.2f} d_depot={d_depot:.2f} "
                      f"best_possible_arrive={best_arrive:.2f} "
                      f"{(best_arrive <= tw_e + 1e-6 and 'SERVABLE') or 'UNSERVABLE'}")
        return super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--inst', type=int, required=True)
    ap.add_argument('--watch', type=str, required=True)  # 逗号分隔 customer id
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num_vehicles', type=int, default=25)
    args = ap.parse_args()

    dataset = dict(np.load(args.data))
    watch = [int(x) for x in args.watch.split(',')]
    rp = DiagReplanner(watch, dataset, args.inst)
    env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles, replanner=rp)
    traces, served_mask = env.run(args.inst)
    print(f"\n=== final: unserved={[c for c in watch if not served_mask[c]]} "
          f"deferred={sorted(rp.deferred_customers)} ===")


if __name__ == '__main__':
    main()
