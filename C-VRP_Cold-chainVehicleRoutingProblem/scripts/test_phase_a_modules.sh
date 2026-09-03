#!/bin/bash
# Test script for Phase A Week 1 modules
# Tests edge_features.py and edge_conditioned_attention.py

set -e

echo "=================================="
echo "Phase A Week 1 Module Tests"
echo "=================================="
echo ""

# Activate environment
if [ -f "/home/hzeng/envs/MASKCO_env/bin/activate" ]; then
    source /home/hzeng/envs/MASKCO_env/bin/activate
    echo "✓ Environment activated (server)"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✓ Using existing conda environment: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  No environment activated, using system Python"
fi

echo ""

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"
echo ""

# Test 1: Edge Features
echo "--- Test 1: Edge Features Module ---"
python scripts/models/edge_features.py
if [ $? -eq 0 ]; then
    echo "✅ Edge features tests passed"
else
    echo "❌ Edge features tests failed"
    exit 1
fi
echo ""

# Test 2: Edge-Conditioned Attention
echo "--- Test 2: Edge-Conditioned Attention ---"
python scripts/models/edge_conditioned_attention.py
if [ $? -eq 0 ]; then
    echo "✅ Edge-conditioned attention tests passed"
else
    echo "❌ Edge-conditioned attention tests failed"
    exit 1
fi
echo ""

# Test 3: Integration test (if both modules work together)
echo "--- Test 3: Integration Test ---"
python - << 'EOF'
import sys
sys.path.insert(0, 'scripts/models')

import jax
import jax.numpy as jnp
import numpy as np

from edge_features import EdgeFeatureExtractor
from edge_conditioned_attention import EdgeConditionedAttention

print("Testing integration of EdgeFeatureExtractor + EdgeConditionedAttention...")

# Setup
B, N, D = 2, 8, 128
coords = jnp.array(np.random.rand(B, N, 2))
tw_start = jnp.array(np.random.rand(B, N) * 10)
tw_end = tw_start + jnp.array(np.random.rand(B, N) * 5 + 5)

# Distance matrix
diff = coords[:, :, None, :] - coords[:, None, :, :]
dist_mat = jnp.sqrt((diff ** 2).sum(axis=-1))

# Initialize modules
extractor = EdgeFeatureExtractor(embed_dim=D)
attention = EdgeConditionedAttention(embed_dim=D, num_heads=8)

# Initialize parameters
extractor_params = extractor.init(
    jax.random.PRNGKey(0), coords, tw_start, tw_end, dist_mat
)
print(f"✓ EdgeFeatureExtractor initialized")

# Compute static edges
static_edges = extractor.apply(
    extractor_params, coords, tw_start, tw_end, dist_mat,
    method=extractor.compute_static_edges
)
print(f"✓ Static edges computed: {static_edges.shape}")

# Initialize attention with edge features
x = jnp.array(np.random.randn(B, N, D).astype(np.float32))
attn_params = attention.init(
    jax.random.PRNGKey(1), x, static_edges, None
)
print(f"✓ EdgeConditionedAttention initialized")

# Forward pass
output = attention.apply(attn_params, x, static_edges, None)
print(f"✓ Attention output: {output.shape}")

assert output.shape == (B, N, D), f"Expected {(B, N, D)}, got {output.shape}"
print(f"✓ Integration test passed!")

EOF

if [ $? -eq 0 ]; then
    echo "✅ Integration test passed"
else
    echo "❌ Integration test failed"
    exit 1
fi
echo ""

echo "=================================="
echo "✅ All Phase A Week 1 tests passed!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Integrate into DynamicColdChainModel"
echo "  2. Run toy instance test (8-node)"
echo "  3. Verify causality and feasibility"
