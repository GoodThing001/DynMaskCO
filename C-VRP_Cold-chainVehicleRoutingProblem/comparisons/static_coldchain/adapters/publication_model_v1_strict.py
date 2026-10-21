"""Reviewed formal-scoring boundary; preserves the original v1 numerical model.

Historical v1 source/results remain untouched. Reject non-finite inputs and
outputs, and prohibit disabling quality constraints in formal solution scoring.
"""
import math

from publication_model_v1 import PublicationModel as OriginalPublicationModel
from publication_model_v1 import CONFIG, RouteResult, load_config, load_original


class PublicationModel(OriginalPublicationModel):
    def route(self, route, enforce_quality=True, departure=None):
        route = tuple(route)
        if departure is not None:
            if isinstance(departure, bool) or not isinstance(departure, (int, float)) or not math.isfinite(departure):
                return RouteResult(route, False, 'nonfinite_or_invalid_departure')
        result = super().route(route, enforce_quality, departure)
        if result.feasible:
            totals = [result.departure, result.returned, result.distance, result.revenue,
                      result.refrigeration_cost, result.quality_loss_value,
                      result.net_benefit, result.min_quality]
            values = totals + [stop[k] for stop in result.stops
                               for k in ['arrival', 'start', 'completion', 'quality', 'wait']]
            if not all(math.isfinite(v) for v in values):
                return RouteResult(route, False, 'nonfinite_evaluation')
        return result

    def solution(self, routes, enforce_quality=True):
        if enforce_quality is not True:
            raise ValueError('formal scoring cannot disable the quality constraint')
        return super().solution(routes, enforce_quality=True)
