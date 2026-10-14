"""核对 3080 GPU 腿 8 个臂是否真的以 cuda 记录落盘（设备口径修正的实测证据）。

动机：`--provider-device cuda` / `--rrnco-device cuda` 只在 provider 真的 `.to(device)` 时才有
实质作用（已核：mvmoe/routefinder/cada 三处 provider 都有 `.to(self.device)` 且张量构造在
self.device 上；rrnco 走 BackendConfig.device）。此工具从**落盘 gate**里把记录的 device 读回来，
避免「脚本改了但实际仍跑 CPU」这种静默失败。

判定用的是**与实际启动命令行对应的那个键**，而不是「gate 里出现过 cuda」：
  - routefinder / cada / mvmoe（`--solver ordering`）→ 看 `provider_device`
  - rrnco（`--solver rrnco`）                        → 看 `rrnco_device`
原因：cada 的 v1 gate 里 `provider_device=cpu` 与另一处 `cuda`（别处的默认值）**同时存在**，
只做「出现过 cuda」的宽松判定会误判为通过。

用法（在 1003 仓库根跑；gate 未回传时用 --root 指向本地暂存目录）：
    python CC_Compare/tools/check_gpu_device.py
    python CC_Compare/tools/check_gpu_device.py --root <dir>
退出码：0 = 8 个臂齐全且全部 cuda；1 = 有缺失或非 cuda。
"""
import argparse
import json
import os
import sys

TAGS = ('mvmoe', 'routefinder', 'cada', 'rrnco')
SUFFIXES = ('0', '51015')
DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'C-VRP_Cold-chainVehicleRoutingProblem', 'results', 'a1_ext_formal2')


def collect_kv(obj, key, out):
    """递归收集指定 key 的所有标量值。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, (str, bool)):
                out.append(str(v))
            elif isinstance(v, (dict, list)):
                collect_kv(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            collect_kv(v, key, out)
    return out


def collect_timeouts(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                collect_timeouts(v, out)
            elif k == 'mean_timeouts' and isinstance(v, (int, float)):
                out.append(float(v))
    elif isinstance(obj, list):
        for v in obj:
            collect_timeouts(v, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--arms', default=None,
                    help='逗号分隔的臂目录名（覆盖默认 4 标签 × 2 档）；'
                         '用于核查 v1 的 `*_f` 终版臂等非默认命名')
    a = ap.parse_args()

    root = os.path.normpath(a.root)
    if a.arms:
        want = [(x.strip(), 'rrnco_device' if x.strip().startswith('rrnco')
                 else 'provider_device') for x in a.arms.split(',') if x.strip()]
    else:
        want = [('%s_%s' % (t, s), 'rrnco_device' if t == 'rrnco' else 'provider_device')
                for t in TAGS for s in SUFFIXES]

    bad, missing, broken = [], [], []
    print('root = %s' % root)
    for arm, key in want:
        p = os.path.join(root, arm, 'gate.json')
        if not os.path.exists(p):
            missing.append(arm)
            print('  %-18s MISSING gate.json' % arm)
            continue
        # 2026-10-13 加固：**空文件/坏 JSON 必须报 FAIL，而不是抛 traceback**。
        # 起因（实测）：收口期间一次错误的上传把 8 个 gate 中的 7 个truncate 成 **0 字节**，
        # 本脚本随即以 `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` **崩溃**。
        # 崩溃会把收口链**在该门禁处直接中断**，后面几道门禁连跑都跑不到；而「报一条 FAIL、
        # 继续跑完」才能让人看到全貌。故此处捕获并归类为 broken。
        try:
            with open(p, encoding='utf-8') as f:
                g = json.load(f)
        except Exception as e:                                   # noqa: BLE001
            sz = os.path.getsize(p)
            broken.append((arm, '%s: %s' % (type(e).__name__, e)))
            print('  %-18s **坏 gate.json（%d 字节）** %s' % (arm, sz, type(e).__name__))
            continue
        vals = sorted(set(collect_kv(g, key, [])))
        to = collect_timeouts(g, [])
        ok = vals == ['cuda']
        print('  %-18s %-16s mean_timeouts=%-8s %s' % (
            arm, '%s=%s' % (key, ','.join(vals) or '-'),
            ('%.2f' % (sum(to) / len(to))) if to else '-',
            'OK' if ok else 'NOT_CUDA'))
        if not ok:
            bad.append((arm, key, vals))

    print()
    if missing or bad or broken:
        if missing:
            print('FAIL: %d 个臂缺 gate.json: %s' % (len(missing), ', '.join(missing)))
        if broken:
            print('FAIL: %d 个臂的 gate.json 无法解析（空文件/坏 JSON ⇒ 必须重传）:' % len(broken))
            for arm, why in broken:
                print('  - %-18s %s' % (arm, why))
        if bad:
            print('FAIL: %d 个臂的实际 device 不是 cuda：' % len(bad))
            for arm, key, vals in bad:
                print('  - %-18s %s=%s' % (arm, key, vals))
        return 1
    print('PASS: %d 个臂齐全，且各自的实际 device 键都是 cuda' % len(want))
    return 0


if __name__ == '__main__':
    sys.exit(main())
