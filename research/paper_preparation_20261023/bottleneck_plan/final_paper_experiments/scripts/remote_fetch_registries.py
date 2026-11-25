# -*- coding: utf-8 -*-
"""抓取三批次 heldout_sources/admission 与 phase，用于构建 P1/P2/P3 注册表。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PROJECT = "/home/hzeng/project/MASKCO-Main"
UNI = PROJECT + "/_maskco_unified_external_20261120"

FILES = {
    "benchmark01_heldout_sources.json": f"{UNI}/benchmark01/heldout_sources.json",
    "benchmark01_heldout_admission.json": f"{UNI}/benchmark01/heldout_admission.json",
    "benchmark01_locked_method.json": f"{UNI}/benchmark01/locked_method.json",
    "prospective01_heldout_sources.json": f"{UNI}/prospective01/heldout_sources.json",
    "prospective01_heldout_admission.json": f"{UNI}/prospective01/heldout_admission.json",
    "retail01_heldout_sources.json": f"{UNI}/retail01/heldout_sources.json",
    "retail01_heldout_admission.json": f"{UNI}/retail01/heldout_admission.json",
    "retail01_locked_method.json": f"{UNI}/retail01/locked_method.json",
    "trajectory01_phase.json": f"{UNI}/trajectory01/phase.json",
    "prospective01_phase.json": f"{UNI}/prospective01/phase.json",
    "benchmark01_phase.json": f"{UNI}/benchmark01/phase.json",
}

LOCAL = HERE.parent / "audit" / "batch_registries"
LOCAL.mkdir(parents=True, exist_ok=True)


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            for name, remote in FILES.items():
                try:
                    s.get(remote, str(LOCAL / name))
                    print("OK", name)
                except Exception as e:
                    print("MISS", name, type(e).__name__)
    finally:
        client.close()


if __name__ == "__main__":
    main()
