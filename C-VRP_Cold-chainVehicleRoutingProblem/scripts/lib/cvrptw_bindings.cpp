#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cvrptw_ops.hpp"

namespace py = pybind11;

static inline int32_t* i32_ptr(py::array_t<int32_t>& a) {
    return static_cast<int32_t*>(a.request().ptr);
}
static inline float* f32_ptr(py::array_t<float>& a) {
    return static_cast<float*>(a.request().ptr);
}

PYBIND11_MODULE(cvrptw_ops, m) {
    m.doc() = "CVRPTW native operators";

    m.def("repair_edd", [](py::array_t<int32_t> routes,
                            py::array_t<float> coords,
                            py::array_t<float> tw_start,
                            py::array_t<float> tw_end,
                            py::array_t<float> service_time,
                            float speed) {
        auto ri = routes.request(), ci = coords.request();
        auto ti = tw_start.request(), tei = tw_end.request(), si = service_time.request();
        cvrptw_repair_edd(i32_ptr(routes), f32_ptr(coords),
                          f32_ptr(tw_start), f32_ptr(tw_end), f32_ptr(service_time),
                          ri.shape[0], ri.shape[1], ci.shape[1], speed);
    }, "EDD repair in-place");

    m.def("two_opt", [](py::array_t<int32_t> routes,
                         py::array_t<float> dist_mat,
                         py::array_t<float> coords,
                         py::array_t<float> tw_start,
                         py::array_t<float> tw_end,
                         py::array_t<float> service_time,
                         int num_steps, int trials_per_step, float speed, int seed) {
        auto ri = routes.request(), di = dist_mat.request(), ci = coords.request();
        auto ti = tw_start.request(), tei = tw_end.request(), si = service_time.request();
        cvrptw_two_opt(i32_ptr(routes), f32_ptr(dist_mat), f32_ptr(coords),
                       f32_ptr(tw_start), f32_ptr(tw_end), f32_ptr(service_time),
                       ri.shape[0], ri.shape[1], ci.shape[1],
                       num_steps, trials_per_step, speed, seed);
    }, "TW-aware 2-opt in-place");

    m.def("count_violations", [](py::array_t<int32_t> routes,
                                  py::array_t<float> coords,
                                  py::array_t<float> tw_start,
                                  py::array_t<float> tw_end,
                                  py::array_t<float> service_time,
                                  float speed) {
        auto ri = routes.request(), ci = coords.request();
        auto ti = tw_start.request(), tei = tw_end.request(), si = service_time.request();
        int B = ri.shape[0], L = ri.shape[1], N = ci.shape[1];
        auto result = py::array_t<int32_t>(B);
        auto* out = i32_ptr(result);
        for (int b = 0; b < B; ++b)
            out[b] = cvrptw_count_violations(
                i32_ptr(routes) + b * L, L,
                f32_ptr(coords) + b * N * 2, N,
                f32_ptr(tw_start) + b * N, f32_ptr(tw_end) + b * N,
                f32_ptr(service_time) + b * N, speed);
        return result;
    }, "Count TW violations per instance");

    // P0-4: 品质感知算子
    m.def("quality_two_opt", [](py::array_t<int32_t> routes,
                                  py::array_t<float> dist_mat,
                                  py::array_t<float> coords,
                                  py::array_t<float> tw_start,
                                  py::array_t<float> tw_end,
                                  py::array_t<float> service_time,
                                  py::array_t<float> quality_loss,
                                  py::array_t<float> energy_mat,
                                  float lambda_q, float lambda_e,
                                  int num_steps, int trials_per_step,
                                  float speed, int seed) {
        auto ri = routes.request(), di = dist_mat.request(), ci = coords.request();
        auto ti = tw_start.request(), tei = tw_end.request(), si = service_time.request();
        auto qi = quality_loss.request(), ei = energy_mat.request();
        cvrptw_quality_two_opt(i32_ptr(routes), f32_ptr(dist_mat), f32_ptr(coords),
                               f32_ptr(tw_start), f32_ptr(tw_end), f32_ptr(service_time),
                               f32_ptr(quality_loss), f32_ptr(energy_mat),
                               lambda_q, lambda_e,
                               ri.shape[0], ri.shape[1], ci.shape[1],
                               num_steps, trials_per_step, speed, seed);
    }, "Quality-aware TW 2-opt in-place");
}
