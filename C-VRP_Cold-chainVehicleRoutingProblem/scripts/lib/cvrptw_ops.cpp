#include "cvrptw_ops.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

static inline float dist2d(float x1, float y1, float x2, float y2) {
    float dx = x1 - x2, dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

int cvrptw_count_violations(
    const int32_t* route, int route_len,
    const float* coords, int num_nodes,
    const float* tw_start, const float* tw_end,
    const float* service_time, float speed)
{
    int violations = 0;
    float cur_time = 0.0f;
    int prev = 0;
    for (int p = 0; p < route_len; ++p) {
        int node = route[p];
        if (node == 0) { cur_time = 0.0f; prev = 0; continue; }
        if (node == prev || node >= num_nodes) continue;
        float d = dist2d(coords[prev * 2], coords[prev * 2 + 1],
                         coords[node * 2], coords[node * 2 + 1]);
        cur_time += service_time[prev] + d / speed;
        if (cur_time < tw_start[node]) cur_time = tw_start[node];
        if (cur_time > tw_end[node] + 1e-6f) ++violations;
        prev = node;
    }
    return violations;
}

// Check if a single route segment is TW-feasible
static bool segment_tw_ok(
    const std::vector<int>& nodes,
    const float* coords, int num_nodes,
    const float* tw_start, const float* tw_end,
    const float* service_time, float speed)
{
    float cur_time = 0.0f;
    int prev = 0;  // depot
    for (int node : nodes) {
        float d = dist2d(coords[prev * 2], coords[prev * 2 + 1],
                         coords[node * 2], coords[node * 2 + 1]);
        cur_time += service_time[prev] + d / speed;
        if (cur_time < tw_start[node]) cur_time = tw_start[node];
        if (cur_time > tw_end[node] + 1e-6f) return false;
        prev = node;
    }
    // return to depot
    float d = dist2d(coords[prev * 2], coords[prev * 2 + 1],
                     coords[0], coords[0 + 1]);
    cur_time += service_time[prev] + d / speed;
    return cur_time <= tw_end[0] + 1e-6f;
}

float cvrptw_route_cost(
    const int32_t* route, int route_len,
    const float* dist_mat, int num_nodes)
{
    float cost = 0.0f;
    int prev = 0;
    for (int p = 0; p < route_len; ++p) {
        int node = route[p];
        if (node == 0) {
            cost += dist_mat[prev * num_nodes + 0];  // return to depot
            prev = 0;
            continue;
        }
        if (node == prev || node >= num_nodes) continue;
        cost += dist_mat[prev * num_nodes + node];
        prev = node;
    }
    if (prev != 0) cost += dist_mat[prev * num_nodes + 0];
    return cost;
}

void cvrptw_repair_edd(
    int32_t* routes, const float* coords, const float* tw_start,
    const float* tw_end, const float* service_time,
    int batch, int route_len, int num_nodes, float speed)
{
    for (int b = 0; b < batch; ++b) {
        auto* route = routes + b * route_len;
        const auto* tw_s = tw_start + b * num_nodes;
        const auto* tw_e = tw_end + b * num_nodes;
        const auto* crd = coords + b * num_nodes * 2;
        const auto* st = service_time + b * num_nodes;

        // Split into segments by depot (0)
        struct Seg { int start, end, len; };
        std::vector<Seg> segs;
        int seg_start = -1;
        for (int p = 0; p < route_len; ++p) {
            if (route[p] == 0) {
                if (seg_start >= 0 && p > seg_start)
                    segs.push_back({seg_start, p, p - seg_start});
                seg_start = -1;
            } else if (seg_start < 0) {
                seg_start = p;
            }
        }
        if (seg_start >= 0 && route_len > seg_start)
            segs.push_back({seg_start, route_len, route_len - seg_start});

        for (auto& seg : segs) {
            if (seg.len <= 1) continue;

            // Collect nodes
            std::vector<int> nodes;
            for (int p = seg.start; p < seg.end; ++p) {
                int n = route[p];
                if (n > 0 && n < num_nodes) nodes.push_back(n);
            }
            if (nodes.empty()) continue;

            // Check if current order is already TW-feasible
            if (segment_tw_ok(nodes, crd, num_nodes, tw_s, tw_e, st, speed))
                continue;

            // Try EDD sort
            std::vector<int> sorted = nodes;
            std::sort(sorted.begin(), sorted.end(),
                      [tw_e](int a, int b) { return tw_e[a] < tw_e[b]; });

            // Only apply if it improves TW
            if (!segment_tw_ok(sorted, crd, num_nodes, tw_s, tw_e, st, speed))
                continue;

            // Write back
            int idx = 0;
            for (int p = seg.start; p < seg.end && idx < (int)sorted.size(); ++p) {
                if (route[p] > 0 && route[p] < num_nodes)
                    route[p] = sorted[idx++];
            }
        }
    }
}

void cvrptw_two_opt(
    int32_t* routes, const float* dist_mat, const float* coords,
    const float* tw_start, const float* tw_end, const float* service_time,
    int batch, int route_len, int num_nodes,
    int num_steps, int trials_per_step, float speed, int seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pos_dist(1, route_len - 2);

    for (int b = 0; b < batch; ++b) {
        auto* sol = routes + b * route_len;
        const auto* dist = dist_mat + b * num_nodes * num_nodes;
        const auto* crd = coords + b * num_nodes * 2;
        const auto* tw_s = tw_start + b * num_nodes;
        const auto* tw_e = tw_end + b * num_nodes;
        const auto* st = service_time + b * num_nodes;

        int best_viol = cvrptw_count_violations(sol, route_len, crd, num_nodes,
                                                  tw_s, tw_e, st, speed);
        float best_cost = cvrptw_route_cost(sol, route_len, dist, num_nodes);

        std::vector<int32_t> tmp(route_len);
        std::vector<int32_t> best_sol(sol, sol + route_len);

        for (int step = 0; step < num_steps; ++step) {
            bool improved = false;
            for (int t = 0; t < trials_per_step; ++t) {
                int i = pos_dist(rng);
                int j = pos_dist(rng);
                if (i >= j) std::swap(i, j);
                if (j - i < 1) continue;
                if (sol[i] == 0 || sol[i + 1] == 0) continue;
                int jj = std::min(j + 1, route_len - 1);
                if (sol[j] == 0 || sol[jj] == 0) continue;

                // 2-opt: reverse [i+1, j]
                std::copy(sol, sol + route_len, tmp.begin());
                for (int k = 0; k <= j - i - 1; ++k)
                    tmp[i + 1 + k] = sol[j - k];

                int new_viol = cvrptw_count_violations(
                    tmp.data(), route_len, crd, num_nodes, tw_s, tw_e, st, speed);
                if (new_viol > best_viol) continue;

                float new_cost = cvrptw_route_cost(tmp.data(), route_len, dist, num_nodes);
                if (new_cost < best_cost - 1e-6f) {
                    std::copy(tmp.begin(), tmp.end(), sol);
                    std::copy(tmp.begin(), tmp.end(), best_sol.begin());
                    best_cost = new_cost;
                    best_viol = new_viol;
                    improved = true;
                    break;
                }
            }
            if (!improved) break;
        }
        std::copy(best_sol.begin(), best_sol.end(), sol);
    }
}

// ============================================================
// P0-4: 冷链品质感知算子
// ============================================================

float cvrptw_quality_cost(
    const int32_t* route, int route_len,
    const float* dist_mat,
    const float* quality_loss,
    const float* energy_mat,
    int num_nodes, float lambda_q, float lambda_e)
{
    float total = 0.0f;
    int prev = 0;
    for (int p = 0; p < route_len; ++p) {
        int node = route[p];
        if (node == 0) {
            total += dist_mat[prev * num_nodes + 0];
            prev = 0;
            continue;
        }
        if (node == prev || node >= num_nodes) continue;
        float d = dist_mat[prev * num_nodes + node];
        float q = quality_loss[node];
        float e = energy_mat[prev * num_nodes + node];
        total += d + lambda_q * q + lambda_e * e;
        prev = node;
    }
    if (prev != 0) {
        total += dist_mat[prev * num_nodes + 0];
        total += lambda_e * energy_mat[prev * num_nodes + 0];
    }
    return total;
}

void cvrptw_quality_two_opt(
    int32_t* routes, const float* dist_mat, const float* coords,
    const float* tw_start, const float* tw_end, const float* service_time,
    const float* quality_loss, const float* energy_mat,
    float lambda_q, float lambda_e,
    int batch, int route_len, int num_nodes,
    int num_steps, int trials_per_step, float speed, int seed)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pos_dist(1, route_len - 2);

    for (int b = 0; b < batch; ++b) {
        auto* sol = routes + b * route_len;
        const auto* dist = dist_mat + b * num_nodes * num_nodes;
        const auto* crd = coords + b * num_nodes * 2;
        const auto* tw_s = tw_start + b * num_nodes;
        const auto* tw_e = tw_end + b * num_nodes;
        const auto* st = service_time + b * num_nodes;
        const auto* ql = quality_loss + b * num_nodes;
        const auto* en = energy_mat + b * num_nodes * num_nodes;

        int best_viol = cvrptw_count_violations(sol, route_len, crd, num_nodes,
                                                  tw_s, tw_e, st, speed);
        float best_cost = cvrptw_quality_cost(sol, route_len, dist, ql, en,
                                               num_nodes, lambda_q, lambda_e);

        std::vector<int32_t> tmp(route_len);
        std::vector<int32_t> best_sol(sol, sol + route_len);

        for (int step = 0; step < num_steps; ++step) {
            bool improved = false;
            for (int t = 0; t < trials_per_step; ++t) {
                int i = pos_dist(rng);
                int j = pos_dist(rng);
                if (i >= j) std::swap(i, j);
                if (j - i < 1) continue;
                if (sol[i] == 0 || sol[i + 1] == 0) continue;
                int jj = std::min(j + 1, route_len - 1);
                if (sol[j] == 0 || sol[jj] == 0) continue;

                std::copy(sol, sol + route_len, tmp.begin());
                for (int k = 0; k <= j - i - 1; ++k)
                    tmp[i + 1 + k] = sol[j - k];

                int new_viol = cvrptw_count_violations(
                    tmp.data(), route_len, crd, num_nodes, tw_s, tw_e, st, speed);
                if (new_viol > best_viol) continue;

                float new_cost = cvrptw_quality_cost(
                    tmp.data(), route_len, dist, ql, en,
                    num_nodes, lambda_q, lambda_e);
                if (new_cost < best_cost - 1e-6f) {
                    std::copy(tmp.begin(), tmp.end(), sol);
                    std::copy(tmp.begin(), tmp.end(), best_sol.begin());
                    best_cost = new_cost;
                    best_viol = new_viol;
                    improved = true;
                    break;
                }
            }
            if (!improved) break;
        }
        std::copy(best_sol.begin(), best_sol.end(), sol);
    }
}
