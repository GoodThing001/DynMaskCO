#!/bin/bash
# 在服务器上直接应用修复的简单脚本

cd ~/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem

echo "应用 Phase A Week 1 __call__ 方法修复..."
echo ""

# 方法 1: 使用 Python 直接插入 __call__ 方法
python3 << 'PYEOF'
import re

# 修复 edge_features.py
print("修复 edge_features.py...")
with open('scripts/models/edge_features.py', 'r') as f:
    content = f.read()

# 检查是否已有 __call__
if 'def __call__' not in content:
    # 在 setup() 方法后插入 __call__
    setup_end = content.find('    def compute_static_edges(')
    if setup_end != -1:
        call_method = '''
    def __call__(
        self,
        coords: jnp.ndarray,
        tw_start: jnp.ndarray,
        tw_end: jnp.ndarray,
        dist_mat: jnp.ndarray,
        visible_mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Default forward pass: compute static edges."""
        return self.compute_static_edges(coords, tw_start, tw_end, dist_mat, visible_mask)

'''
        content = content[:setup_end] + call_method + content[setup_end:]
        with open('scripts/models/edge_features.py', 'w') as f:
            f.write(content)
        print("✓ edge_features.py 已添加 __call__ 方法")
    else:
        print("⚠ 无法找到插入位置")
else:
    print("✓ edge_features.py 已有 __call__ 方法")

# 修复 DynamicColdChainModelEdgeState.py
print("\n修复 DynamicColdChainModelEdgeState.py...")
with open('scripts/models/DynamicColdChainModelEdgeState.py', 'r') as f:
    content = f.read()

if 'def __call__' not in content or content.count('def __call__') < 2:
    # 在 setup() 方法后插入 __call__
    # 找到 setup 方法的结束位置（下一个 def）
    setup_match = re.search(r'(    def setup\(self\):.*?)(    def \w+\()', content, re.DOTALL)
    if setup_match:
        call_method = '''
    def __call__(
        self,
        x: jnp.ndarray,
        visible_mask: jnp.ndarray,
        coords: Optional[jnp.ndarray] = None,
        tw_start: Optional[jnp.ndarray] = None,
        tw_end: Optional[jnp.ndarray] = None,
        dist_mat: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Default forward pass: encode with edge features."""
        return self.encode(x, visible_mask, coords, tw_start, tw_end, dist_mat)

'''
        content = setup_match.group(1) + call_method + setup_match.group(2)
        content = re.sub(r'(    def setup\(self\):.*?)(    def \w+\()',
                        lambda m: m.group(1) + call_method + m.group(2),
                        content, count=1, flags=re.DOTALL)
        with open('scripts/models/DynamicColdChainModelEdgeState.py', 'w') as f:
            f.write(content)
        print("✓ DynamicColdChainModelEdgeState.py 已添加 __call__ 方法")
    else:
        print("⚠ 无法找到插入位置")
else:
    print("✓ DynamicColdChainModelEdgeState.py 已有 __call__ 方法")

print("\n✅ 所有修复已应用")
PYEOF

echo ""
echo "现在可以重新运行测试:"
echo "  bash scripts/test_phase_a_complete.sh 2>&1 | tee /tmp/phase_a_test.log"
