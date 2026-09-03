#!/bin/bash
# 快速修复 Flax 模块的 __call__ 方法问题

set -e

echo "应用 Phase A Week 1 模块修复..."
echo ""

# 修复 edge_features.py
echo "修复 1/2: edge_features.py"
cat > /tmp/edge_features_fix.patch << 'EOF'
--- a/scripts/models/edge_features.py
+++ b/scripts/models/edge_features.py
@@ -32,6 +32,17 @@ class EdgeFeatureExtractor(nn.Module):
             name='edge_head_proj'
         )

+    def __call__(
+        self,
+        coords: jnp.ndarray,
+        tw_start: jnp.ndarray,
+        tw_end: jnp.ndarray,
+        dist_mat: jnp.ndarray,
+        visible_mask: Optional[jnp.ndarray] = None
+    ) -> jnp.ndarray:
+        """Default forward pass: compute static edges."""
+        return self.compute_static_edges(coords, tw_start, tw_end, dist_mat, visible_mask)
+
     def compute_static_edges(
         self,
         coords: jnp.ndarray,      # [B, N, 2] (x, y coordinates)
EOF

# 在 edge_features.py 中的 setup() 方法后添加 __call__
sed -i '/def setup(self):/,/edge_head_proj/a\
\
    def __call__(\
        self,\
        coords: jnp.ndarray,\
        tw_start: jnp.ndarray,\
        tw_end: jnp.ndarray,\
        dist_mat: jnp.ndarray,\
        visible_mask: Optional[jnp.ndarray] = None\
    ) -> jnp.ndarray:\
        """Default forward pass: compute static edges."""\
        return self.compute_static_edges(coords, tw_start, tw_end, dist_mat, visible_mask)' scripts/models/edge_features.py

echo "✓ edge_features.py 已修复"

# 修复 DynamicColdChainModelEdgeState.py
echo "修复 2/2: DynamicColdChainModelEdgeState.py"

# 在 setup() 方法后添加 __call__
sed -i '/self.encoder_layers = \[/,/\]/a\
\
    def __call__(\
        self,\
        x: jnp.ndarray,\
        visible_mask: jnp.ndarray,\
        coords: Optional[jnp.ndarray] = None,\
        tw_start: Optional[jnp.ndarray] = None,\
        tw_end: Optional[jnp.ndarray] = None,\
        dist_mat: Optional[jnp.ndarray] = None\
    ) -> jnp.ndarray:\
        """Default forward pass: encode with edge features."""\
        return self.encode(x, visible_mask, coords, tw_start, tw_end, dist_mat)' scripts/models/DynamicColdChainModelEdgeState.py

echo "✓ DynamicColdChainModelEdgeState.py 已修复"
echo ""
echo "✅ 所有修复已应用"
echo ""
echo "现在可以重新运行测试:"
echo "  bash scripts/test_phase_a_complete.sh"
