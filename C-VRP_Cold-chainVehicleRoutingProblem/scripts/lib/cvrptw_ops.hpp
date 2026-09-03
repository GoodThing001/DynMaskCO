#pragma once
#include <vector>
#include <cstdint>

// EDD 修复：对每条子路径按交货期限排序
void cvrptw_repair_edd(
    int32_t* routes,         // [batch, route_len]
    const float* coords,     // [batch, num_nodes, 2]
    const float* tw_start,   // [batch, num_nodes]
    const float* tw_end,     // [batch, num_nodes]
    const float* service_time,// [batch, num_nodes]
    int batch, int route_len, int num_nodes, float speed
);

// 单实例 TW 可行性检查，返回违规次数
int cvrptw_count_violations(
    const int32_t* route, int route_len,
    const float* coords, int num_nodes,
    const float* tw_start, const float* tw_end,
    const float* service_time, float speed
);

// 单实例路径成本
float cvrptw_route_cost(
    const int32_t* route, int route_len,
    const float* dist_mat, int num_nodes
);

// TW-aware 2-opt（随机采样）
void cvrptw_two_opt(
    int32_t* routes,         // [batch, route_len]
    const float* dist_mat,   // [batch, num_nodes, num_nodes]
    const float* coords,     // [batch, num_nodes, 2]
    const float* tw_start,   // [batch, num_nodes]
    const float* tw_end,     // [batch, num_nodes]
    const float* service_time,// [batch, num_nodes]
    int batch, int route_len, int num_nodes,
    int num_steps, int trials_per_step, float speed, int seed
);

// === P0-4: 冷链品质感知 ===

// 品质感知成本: total = distance + lambda_q * quality_loss + lambda_e * energy
float cvrptw_quality_cost(
    const int32_t* route, int route_len,
    const float* dist_mat,
    const float* quality_loss,  // [num_nodes] per-node Arrhenius decay
    const float* energy_mat,    // [num_nodes, num_nodes] refrigeration energy
    int num_nodes, float lambda_q, float lambda_e
);

// 品质感知 2-opt: swap 接受条件包含 quality_cost
void cvrptw_quality_two_opt(
    int32_t* routes,         // [batch, route_len]
    const float* dist_mat,   // [batch, num_nodes, num_nodes]
    const float* coords,     // [batch, num_nodes, 2]
    const float* tw_start,   // [batch, num_nodes]
    const float* tw_end,     // [batch, num_nodes]
    const float* service_time,// [batch, num_nodes]
    const float* quality_loss,// [batch, num_nodes]
    const float* energy_mat, // [batch, num_nodes, num_nodes]
    float lambda_q, float lambda_e,
    int batch, int route_len, int num_nodes,
    int num_steps, int trials_per_step, float speed, int seed
);
