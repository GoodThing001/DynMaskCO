# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in the **DynMaskCO-CC extension workspace** (`C-VRP_Cold-chainVehicleRoutingProblem/`).

> **Authoritative guidance is at the repo root [`../CLAUDE.md`](../CLAUDE.md)** — read it first for project overview, current status, hard invariants, common commands, architecture, and environment. This file only holds extension-specific notes **not** repeated there.

## Extension-specific hard constraints

- Work inside this directory only; **never modify `../MASKCO_code/`**.
- **No `__init__.py`** in extension subdirectories — imports go through `sys.path.insert`.
- **Frozen `data/baseline/` files must not be modified.**
- The extension has C++ extensions in `scripts/lib/` (EDD repair + TW-aware 2-opt); after editing the `.cpp`/`.hpp`, rebuild with `cd scripts/lib && make`.
- Path resolution: `scripts/project_paths.py` resolves `WORKSPACE_ROOT`, `MASKCO_ROOT`, and the extension root — prefer it over hard-coded paths.

## Authoritative status

The root `../CLAUDE.md` "Current Status" is a snapshot, not the authority. The single authoritative status entry is [`项目当前状态.md`](项目当前状态.md) + [`docs/当前规划/结果与进度/执行进度表.md`](docs/当前规划/结果与进度/执行进度表.md). When research status conflicts with the root CLAUDE.md, those win.
