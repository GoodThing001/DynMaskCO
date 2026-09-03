"""CaDA torchrl 新旧 API 兼容补丁（在 import envs.env 之前执行）。

CaDA 原始代码（torchrl 0.1.x）使用旧类名:
    CompositeSpec / BoundedTensorSpec / UnboundedContinuousTensorSpec /
    UnboundedDiscreteTensorSpec / EnvBase(from torchrl.envs)
现代 torchrl(>=0.6) 已重命名为 Composite / Bounded / Unbounded 等，EnvBase 移至
torchrl.envs.common。本补丁在旧名字缺失时把它们指向新名字（幂等，不覆盖已存在者）。

必须在 `from dcc_env import DCCEnv`（进而 import CaDA 的 envs.env）之前调用。
"""
import torchrl.data as _td
import torchrl.envs as _te

_SPEC_RENAMES = [
    ("CompositeSpec", "Composite"),
    ("BoundedTensorSpec", "Bounded"),
    ("UnboundedContinuousTensorSpec", "UnboundedContinuous"),
    ("UnboundedDiscreteTensorSpec", "UnboundedDiscrete"),
    # 兜底：新版可能只有统一的 Unbounded
]

for _old, _new in _SPEC_RENAMES:
    if not hasattr(_td, _old):
        if hasattr(_td, _new):
            setattr(_td, _old, getattr(_td, _new))
        elif _new in ("UnboundedContinuous", "UnboundedDiscrete") and hasattr(_td, "Unbounded"):
            setattr(_td, _old, getattr(_td, "Unbounded"))

# EnvBase：新版移到 torchrl.envs.common
if not hasattr(_te, "EnvBase"):
    try:
        from torchrl.envs.common import EnvBase as _EnvBase
        _te.EnvBase = _EnvBase
    except Exception:
        pass
