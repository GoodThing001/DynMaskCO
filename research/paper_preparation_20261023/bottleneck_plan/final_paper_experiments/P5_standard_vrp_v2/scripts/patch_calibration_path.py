import ast
from pathlib import Path

p = Path("standard_vrp/core/calibration.py")
t = p.read_text(encoding="utf-8")

old = '''LOCKED_CALIBRATED = Path(r"D:\\PyCharm_\\MASKCO-Main\\research\\paper_preparation_20261023"
                         r"\\bottleneck_plan\\maskco_perishable_external_20261119"
                         r"\\reserved_validation\\solver\\calibrated.json")


def locked_alpha():
    """读取锁版冻结 alpha。"""
    data = json.loads(LOCKED_CALIBRATED.read_text(encoding="utf-8"))
    return float(data["alpha"])'''
new = '''# 解锁版 alpha 的三条来源（按优先级）：包内副本 → 环境变量 → 仓库只读路径。
# 包内副本是锁版 calibrated.json 的逐字节拷贝（SHA 见 manifests/frozen_identity.json），
# 保证远端部署自足；任何来源都读不到时**直接报错**，不得静默回退（D8 教训）。
PACKAGED = Path(__file__).resolve().parent / "calibrated.json"
REPO_LOCKED = Path(r"D:\\PyCharm_\\MASKCO-Main\\research\\paper_preparation_20261023"
                   r"\\bottleneck_plan\\maskco_perishable_external_20261119"
                   r"\\reserved_validation\\solver\\calibrated.json")
ENV_KEY = "MASKCO_P5V2_CALIBRATED"
LOCKED_CALIBRATED = PACKAGED


def calibrated_path():
    import os
    for cand in (PACKAGED, Path(os.environ[ENV_KEY]) if os.environ.get(ENV_KEY) else None,
                 REPO_LOCKED):
        if cand is not None and Path(cand).exists():
            return Path(cand)
    raise FileNotFoundError(
        "locked calibrated.json not found (packaged/env/repo); 校准热图不得静默缺失")


def locked_alpha():
    """读取锁版冻结 alpha（缺失即报错）。"""
    data = json.loads(calibrated_path().read_text(encoding="utf-8"))
    return float(data["alpha"])'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
ast.parse(t)

# pool_search：校准失败不再静默
q = Path("standard_vrp/solver/pool_search.py")
t2 = q.read_text(encoding="utf-8")
old2 = '''        heat_cal = None
        if heat is not None and mode_name in ("hybrid", "choice"):
            try:
                heat_cal = calibrated_heat(dist, heat)
            except Exception:
                heat_cal = None'''
new2 = '''        heat_cal = None
        if heat is not None and mode_name in ("hybrid", "choice"):
            try:
                heat_cal = calibrated_heat(dist, heat)
            except Exception as exc:  # noqa: BLE001
                # 校准失败必须留痕（D8：曾因硬编码路径 + 静默回退导致校准从未生效）
                pred.note_error(exc, "calibration")
                fallbacks += 1'''
assert old2 in t2
t2 = t2.replace(old2, new2)
q.write_text(t2, encoding="utf-8")
ast.parse(t2)
print("calibration path fixed + silent fallback removed")
