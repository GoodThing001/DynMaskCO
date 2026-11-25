#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace {
using Route = std::vector<int>;
using Solution = std::vector<Route>;
using Clock = std::chrono::steady_clock;
constexpr double eps = 1e-7;
constexpr double bad = -std::numeric_limits<double>::infinity();
constexpr int customers = 100;

struct Economy {
    const double *nodes, *dist, *p;
    long long calls = 0;
    double node(int i, int k) const { return nodes[7 * i + k]; }
    double arc(int i, int j) const { return dist[101 * i + j]; }
    double value(const Route& route) {
        ++calls;
        if (route.empty()) return 0.;
        double mass = 0, distance = 0, prefix = 0, depart = node(0, 5);
        int prev = 0;
        for (int i : route) {
            mass += node(i, 3);
            double link = arc(prev, i);
            distance += link;
            prefix += link / p[0];
            depart = std::min(depart, node(i, 5) - prefix);
            prefix += node(i, 6);
            prev = i;
        }
        if (mass > p[6] + eps) return bad;
        distance += arc(prev, 0);
        depart = std::min(depart, node(0, 5) - prefix - arc(prev, 0) / p[0]);
        if (depart < node(0, 4) - eps) return bad;
        double at = depart, revenue = 0;
        prev = 0;
        for (int i : route) {
            at = std::max(at + arc(prev, i) / p[0], node(i, 4));
            if (at > node(i, 5) + eps) return bad;
            at += node(i, 6);
            double quality = 1 - p[1] * (at - depart);
            if (quality < p[2] - eps) return bad;
            revenue += p[3] * node(i, 3) * quality;
            prev = i;
        }
        at += arc(prev, 0) / p[0];
        if (at > node(0, 5) + eps) return bad;
        return revenue - p[4] * distance - p[5] * (at - depart);
    }
    double total(const Solution& routes) {
        double sum = 0;
        for (const Route& r : routes) sum += value(r);
        return sum;
    }
};

Solution unpack(const int* buffer) {
    Solution out(1);
    for (; *buffer != -1; ++buffer) {
        if (*buffer == 0) out.emplace_back();
        else out.back().push_back(*buffer);
    }
    return out;
}
void pack(const Solution& routes, int* buffer) {
    for (size_t i = 0; i < routes.size(); ++i) {
        if (i) *buffer++ = 0;
        for (int j : routes[i]) *buffer++ = j;
    }
    *buffer = -1;
}

struct Slot { int route, pos; double detour, learned; };
struct Move { int route, pos; double gain; };
struct Suggestion { Move best; double regret; };

Suggestion choose_slot(Economy& e, const double* heat, Solution& routes,
                       const std::vector<double>& values, int i, bool regret) {
    std::vector<Slot> slots;
    for (int r = 0; r < int(routes.size()); ++r) {
        const auto& route = routes[r];
        double mass = e.node(i, 3);
        for (int v : route) mass += e.node(v, 3);
        if (mass > e.p[6] + eps) continue;
        for (int pos = 0; pos <= int(route.size()); ++pos) {
            int a = pos ? route[pos-1] : 0, b = pos < int(route.size()) ? route[pos] : 0;
            double detour = e.arc(a,i) + e.arc(i,b) - e.arc(a,b);
            double h = heat[101*a+i] + heat[101*i+b] - heat[101*a+b];
            slots.push_back({r,pos,detour,h});
        }
    }
    auto distance_order = [](const Slot& a, const Slot& b) {
        if (a.detour != b.detour) return a.detour < b.detour;
        return std::pair<int,int>(a.route,a.pos) < std::pair<int,int>(b.route,b.pos);
    };
    auto neural_order = [](const Slot& a, const Slot& b) {
        if (a.learned != b.learned) return a.learned > b.learned;
        return std::pair<int,int>(a.route,a.pos) < std::pair<int,int>(b.route,b.pos);
    };
    Move best{int(routes.size()),0,e.value(Route{i})};
    bool feasible = false;
    double second = bad;
    auto value_at = [&](const Slot& s) {
        auto& r = routes[s.route];
        r.insert(r.begin()+s.pos, i);
        double value = e.value(r);
        r.erase(r.begin()+s.pos);
        return value-values[s.route];
    };
    auto inspect = [&](const Slot& s) {
        double gain=value_at(s);
        if (!std::isfinite(gain)) return;
        feasible = true;
        if (gain > best.gain + 1e-9 || (std::abs(gain-best.gain) <= 1e-9 &&
            std::pair<int,int>(s.route,s.pos) < std::pair<int,int>(best.route,best.pos))) {
            second=best.gain;
            best = {s.route,s.pos,gain};
        } else second=std::max(second,gain);
    };
    if (regret) {
        int end=std::min(8,int(slots.size()));
        std::partial_sort(slots.begin(),slots.begin()+end,slots.end(),neural_order);
        std::vector<Slot> selected(slots.begin(),slots.begin()+end);
        if (int(slots.size())>end) {
            std::sort(slots.begin(),slots.end(),distance_order);
            const Slot* first=nullptr;
            bool protected_slot=false;
            for (const Slot& s:slots) {
                if (first && s.detour>first->detour) break;
                if (!std::isfinite(value_at(s))) continue;
                if (!first) first=&s;
                for (const Slot& t:selected) {
                    if (t.route==s.route && t.pos==s.pos) protected_slot=true;
                }
                if (protected_slot) break;
            }
            if (first && !protected_slot) selected.back()=*first;
        }
        for (const Slot& s:selected) inspect(s);
    } else {
        int keep = std::min(6,int(slots.size()));
        std::partial_sort(slots.begin(), slots.begin()+keep, slots.end(),distance_order);
        int end=std::min(keep+6,int(slots.size()));
        std::partial_sort(slots.begin()+keep,slots.begin()+end,slots.end(),neural_order);
        for (int k=0;k<end;++k) inspect(slots[k]);
        if (!feasible) {
            std::sort(slots.begin()+end,slots.end(),distance_order);
            for (size_t k=end;k<slots.size() && !feasible;++k) inspect(slots[k]);
        }
    }
    return {best,std::isfinite(second)?best.gain-second:best.gain};
}

void insert_customer(Economy& e,Solution& routes,std::vector<double>& values,int i,Move best) {
    if (best.route == int(routes.size())) {
        routes.push_back({i});
        values.push_back(best.gain);
    } else {
        auto& r = routes[best.route];
        r.insert(r.begin()+best.pos,i);
        values[best.route] = e.value(r);
    }
}

Solution reconstruct(Economy& e, const Solution& parent, const Route& mask, const double* heat,bool regret) {
    std::array<bool,101> hidden{};
    for (int i : mask) hidden[i] = true;
    Solution rebuilt;
    std::vector<double> values;
    for (const auto& r : parent) {
        Route part;
        for (int v : r) if (!hidden[v]) part.push_back(v);
        if (!part.empty()) {
            double u = e.value(part);
            if (!std::isfinite(u)) return parent;
            rebuilt.push_back(part);
            values.push_back(u);
        }
    }
    if (regret) {
        Route pending=mask;
        std::sort(pending.begin(),pending.end());
        while (!pending.empty()) {
            double highest=bad;
            size_t winner=0;
            Move chosen{};
            for (size_t j=0;j<pending.size();++j) {
                Suggestion s=choose_slot(e,heat,rebuilt,values,pending[j],true);
                if (s.regret>highest) {highest=s.regret;winner=j;chosen=s.best;}
            }
            insert_customer(e,rebuilt,values,pending[winner],chosen);
            pending.erase(pending.begin()+winner);
        }
    } else for (int i:mask) insert_customer(e,rebuilt,values,i,choose_slot(e,heat,rebuilt,values,i,false).best);
    return rebuilt;
}
}

#ifdef _WIN32
#define EXPORTED extern "C" __declspec(dllexport)
#else
#define EXPORTED extern "C"
#endif

EXPORTED double route_score(const double* nodes, const double* dist, const double* p,
                            const int* order, int n) {
    Economy economy{nodes,dist,p};
    return economy.value(Route(order,order+n));
}

EXPORTED void repair_fixed(const double* nodes,const double* dist,const double* p,const double* heat,
                           const int* mask,int count,const int* parent,int* answer) {
    Economy economy{nodes,dist,p};
    pack(reconstruct(economy,unpack(parent),Route(mask,mask+count),heat,true),answer);
}

// Each batch shares one masked network prediction. Both search states are returned.
EXPORTED void repair_batch(const double* nodes, const double* dist, const double* p,
    const double* heat, const int* group, int group_size, int* current, int* best,
    unsigned seed, double seconds, double tolerance, int related, int regret, int iteration_cap, long long* stats) {
    const auto deadline = Clock::now() + std::chrono::duration<double>(seconds);
    Economy economy{nodes,dist,p};
    Solution state = unpack(current), record = unpack(best);
    double u = economy.total(state), best_u = economy.total(record);
    std::mt19937 rng(seed);
    Route order(group,group+group_size);
    std::uniform_real_distribution<double> random(0,1);
    int iterations = 0, improvements = 0;
    while (Clock::now() < deadline && (iteration_cap<=0 || iterations<iteration_cap)) {
        std::shuffle(order.begin(),order.end(),rng);
        int k = std::min(group_size, regret ? 10 : 6 + int(rng()%12));
        Route mask(order.begin(),order.begin()+k);
        if (related && iterations%2 == 0) {
            int anchor = order[0];
            std::vector<std::pair<double,int>> priorities;
            for (int i : order) if (i != anchor) {
                double affinity = heat[anchor*101+i] + heat[i*101+anchor];
                priorities.emplace_back(affinity,i);
            }
            std::sort(priorities.begin(),priorities.end(),std::greater<std::pair<double,int>>());
            mask = {anchor};
            for (int j=0; j<k-1; ++j) {
                size_t pos = size_t(random(rng)*random(rng)*priorities.size());
                mask.push_back(priorities[pos].second);
                priorities.erase(priorities.begin()+pos);
            }
            std::shuffle(mask.begin(),mask.end(),rng);
        }
        Solution next = reconstruct(economy,state,mask,heat,regret!=0);
        double new_u = economy.total(next);
        if (new_u > best_u + eps) {
            record = next; best_u = new_u; ++improvements;
        }
        if (new_u >= u-eps || (tolerance > 0 && new_u > best_u - 6*tolerance &&
                                random(rng) < std::exp((new_u-u)/tolerance))) {
            state = std::move(next); u = new_u;
        }
        ++iterations;
    }
    pack(state,current); pack(record,best);
    stats[0]=iterations; stats[1]=improvements; stats[2]=economy.calls;
}
