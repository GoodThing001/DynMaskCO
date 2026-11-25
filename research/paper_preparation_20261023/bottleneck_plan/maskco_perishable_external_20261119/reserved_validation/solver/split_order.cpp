#include "masked_repair.cpp"

// Longest path on the acyclic graph of feasible consecutive route segments.
EXPORTED double split_order(const double* nodes, const double* dist, const double* p,
                            const int* order, int n, int* previous, long long* calls) {
    Economy economy{nodes, dist, p};
    std::array<double,customers+1> best{};
    best.fill(bad);
    best[0] = 0.;
    for (int start=0;start<n;++start) {
        if (!std::isfinite(best[start])) continue;
        Route segment;
        segment.reserve(n-start);
        for (int stop=start;stop<n;++stop) {
            segment.push_back(order[stop]);
            double value=best[start]+economy.value(segment);
            if (value>best[stop+1]+eps) {
                best[stop+1]=value;
                previous[stop+1]=start;
            }
        }
    }
    *calls=economy.calls;
    return best[n];
}
