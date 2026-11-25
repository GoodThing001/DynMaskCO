import json
import os
from config import DATA_PATH, FILES


VARIANT_FEATURES = {
    "VRP": frozenset(),
    "HVRP": frozenset({"HVRP"}),
    "MTVRP": frozenset({"MTVRP"}),
    "TDVRP": frozenset({"TDVRP"}),
    "HMTTDVRP": frozenset({"HVRP", "MTVRP", "TDVRP"}),
    "HSIDVRP": frozenset({"HVRP", "SIDVRP"}),
    "SDVRP": frozenset({"SDVRP"}),
    "MTVRPDB": frozenset({"MTVRP", "VRPDB"}),
    "RWVRP": frozenset({"HVRP", "MTVRP", "TDVRP", "SIDVRP", "SDVRP", "VRPDB"}),
}


def validate_solutions():
    city = "DummyCity"
    test_solution_paths = ["VRP.json", "HVRP.json", "MTVRP.json", "TDVRP.json", "HMTTDVRP.json", "HSIDVRP.json", "SDVRP.json", 
                           "MTVRPDB.json", "RWVRP.json", "HSIDVRP_inaccessible.json", "MTVRPDB_missing_break.json"]
    for test_sol in test_solution_paths:
        print('===============' + test_sol + '===============')
        solution_path = os.path.join(DATA_PATH, city, "solutions", test_sol)
        solution = load(solution_path)
        print(json.dumps(validate(solution, city, print_routes=True), indent=2))


def load(path):
    with open(path) as f:
        return json.load(f)


def load_instance(city):
    city_dir = os.path.join(DATA_PATH, city)
    locations = load(os.path.join(city_dir, FILES["locations"]))
    vehicles = load(os.path.join(city_dir, FILES["vehicles"]))
    time_matrix = load(os.path.join(city_dir, FILES["time_matrix_static"]))
    time_matrix_tdvrp = load(os.path.join(city_dir, FILES["time_matrix_tdvrp"]))
    return locations, vehicles, time_matrix, time_matrix_tdvrp


def _time_to_sec(t):
    h, m = map(int, t.split(":")); return h * 3600 + m * 60


def _get_travel(matrix, departure):
    times = sorted((_time_to_sec(k), v["matrix_seconds"]) for k, v in matrix["matrices"].items())
    selected = times[0][1]
    for start, mat in times:
        if departure >= start: selected = mat
    return selected


def _fail(vehicle_id, location_id, reason):
    return {"feasible": False, "vehicle_id": vehicle_id, "location_id": location_id, "reason": reason}


def print_route_summary(route_summaries):
    for vehicle_id, stops, breaks in route_summaries:
        steps = []
        for index, (location_id, quantity, start, end) in enumerate(stops):
            steps.append(f"{location_id} ({quantity:g}) [{start:g}-{end:g}]")
            if index in breaks:
                break_start, break_end = breaks[index]
                steps.append(f"BREAK [{break_start:g}-{break_end:g}]")
        print(f"Vehicle {vehicle_id}: {' -> '.join(steps)}")


def _service_time(location, quantity, is_sdvrp):
    if is_sdvrp and location["id"] != 0:
        rates = {"supermarket": 120, "convenience": 90}
        rate = rates.get(location["location_type"])
        if rate is not None:
            return int(5 * 60 + quantity * rate)
        return location.get("service_time_sdvrp", location["service_time"])
    return location["service_time"]


def validate(solution, city, print_routes=False):
    """Validate routes; SDVRP unloading is 5 minutes plus 120/90 seconds per unit
    delivered at supermarkets/convenience stores, respectively.
    """
    locations_data, vehicles_data, static_matrix, tdvrp_matrix = load_instance(city)
    locations = {x["id"]: x for x in locations_data}
    vehicles = {x["id"]: x for x in vehicles_data}

    variant = solution.get("instance_name", "").rsplit("-", 1)[-1].upper()
    features = VARIANT_FEATURES.get(variant)
    if features is None:
        return _fail(None, None, f"unknown instance variant: {variant}")

    default_matrix = tdvrp_matrix if "TDVRP" in features else static_matrix
    matrix_cache = {}
    accessibility = None
    access_rows = {}
    if "SIDVRP" in features:
        accessibility = load(os.path.join(DATA_PATH, city, FILES["vehicle_location_accessibility"]))
        access_rows = {
            vehicle["id"]: index
            for index, vehicle in enumerate(accessibility["vehicles"])
        }

    total_time_consumed = 0
    driving_time = 0
    delivered = {location_id: 0 for location_id in locations if location_id != 0}
    served_locations = set()
    used_vehicles = set()
    route_summaries = []

    for r in solution["routes"]:
        vid = r["vehicle_id"]
        if vid not in vehicles: 
            return _fail(vid, None, "unknown vehicle")
        if vid in used_vehicles:
            return _fail(vid, None, "vehicle may only have one route entry")
        used_vehicles.add(vid)

        vehicle = vehicles[vid]
        route = r["route"]
        demands = r.get("demands")
        breaks_after = r.get("breaks_after", [])

        if "HVRP" in features:
            if "hvrp_capacity" not in vehicle:
                return _fail(vid, None, "vehicle is missing hvrp_capacity")
            capacity_class = vehicle["hvrp_capacity"]
            matrix_key = ("TDVRP" in features, capacity_class)
            if matrix_key not in matrix_cache:
                template_key = (
                    "time_matrix_tdvrp_hvrp"
                    if "TDVRP" in features
                    else "time_matrix_hvrp"
                )
                filename = FILES[template_key].format(capacity=capacity_class)
                matrix_path = os.path.join(DATA_PATH, city, filename)
                try:
                    matrix_cache[matrix_key] = load(matrix_path)
                except FileNotFoundError:
                    return _fail(
                        vid,
                        None,
                        f"missing travel matrix for HVRP capacity {capacity_class}",
                    )
            vehicle_matrix = matrix_cache[matrix_key]
        else:
            vehicle_matrix = default_matrix

        if not route: 
            return _fail(vid, None, "empty route")
        if len(route) < 2:
            return _fail(vid, None, "route must contain a depot departure and return")
        if route[0] != 0 or route[-1] != 0: 
            return _fail(vid, route[0], "route must start and end at depot")
        if demands is not None and len(demands) != len(route): 
            return _fail(vid, None, "route and demands must have the same length")
        if not isinstance(breaks_after, list):
            return _fail(vid, None, "breaks_after must be a list of route indexes")
        if breaks_after and "VRPDB" not in features:
            return _fail(vid, None, "breaks are only allowed for VRPDB")
        if any(not isinstance(index, int) or isinstance(index, bool) for index in breaks_after):
            return _fail(vid, None, "break indexes must be integers")
        if len(set(breaks_after)) != len(breaks_after):
            return _fail(vid, None, "duplicate break index")
        if any(
            index <= 0 or index >= len(route) - 1
            or route[index] == 0
            for index in breaks_after
        ):
            return _fail(vid, None, "break index must identify a delivery stop before the final depot")

        if "MTVRP" not in features and 0 in route[1:-1]:
            return _fail(vid, 0, "intermediate depot visits require MTVRP")
        if any(route[index] == 0 and route[index - 1] == 0 for index in range(1, len(route))):
            return _fail(vid, 0, "consecutive depot visits are not allowed")

        time = vehicle["driver_tw"][0]
        load_now = 0
        driving_since_break = 0
        capacity_key = "hvrp_capacity" if "HVRP" in features else "capacity"
        capacity = vehicle[capacity_key]
        driving_limit = vehicle.get("driving_before_break_mins", 0) * 60
        break_duration = vehicle.get("break_mins", 0) * 60
        route_driving_time = 0
        route_stops = [(route[0], 0, time, time)]
        time += _service_time(locations[route[0]], 0, False)
        route_stops[0] = (*route_stops[0][:3], time)
        route_breaks = {}

        for i in range(len(route) - 1):
            loc_id = route[i]
            next_id = route[i + 1]
            if loc_id not in locations: 
                return _fail(vid, loc_id, "unknown location")
            if next_id not in locations: 
                return _fail(vid, next_id, "unknown location")

            if accessibility is not None:
                vehicle_row = access_rows.get(vid)
                access_matrix = accessibility["accessibility_matrix"]
                if (
                    vehicle_row is None
                    or vehicle_row >= len(access_matrix)
                    or loc_id >= len(access_matrix[vehicle_row])
                    or next_id >= len(access_matrix[vehicle_row])
                    or not access_matrix[vehicle_row][loc_id]
                    or not access_matrix[vehicle_row][next_id]
                ):
                    return _fail(vid, next_id, "vehicle cannot access location")

            if "VRPDB" in features and i in breaks_after:
                break_start = time
                time += break_duration
                driving_since_break = 0
                route_breaks[i] = (break_start, time)
            route_stops[-1] = (*route_stops[-1][:3], break_start if i in route_breaks else time)

            if "TDVRP" in features:
                mat = _get_travel(vehicle_matrix, time)
                travel = mat[loc_id][next_id]
            else:
                travel = vehicle_matrix["matrix_seconds"][loc_id][next_id]

            if "VRPDB" in features:
                if travel > driving_limit:
                    return _fail(vid, next_id, "single drive exceeds driving limit before break")
                if driving_since_break + travel > driving_limit:
                    return _fail(vid, next_id, "required break missing from breaks_after")

            time += travel
            route_driving_time += travel
            driving_since_break += travel
            arrival_time = time
            next_loc = locations[next_id]
            tw_start, tw_end = next_loc["time_window"]
            if time < tw_start: time = tw_start
            if time > tw_end: 
                return _fail(vid, next_id, "time window violated")

            if next_id != 0:
                if "SDVRP" in features:
                    if demands is None:
                        return _fail(vid, next_id, "demands are required for SDVRP")
                    quantity = demands[i + 1]
                    if not isinstance(quantity, (int, float)) or isinstance(quantity, bool):
                        return _fail(vid, next_id, "delivery quantity must be a number")
                else:
                    quantity = next_loc["demand"]
                if quantity < 0:
                    return _fail(vid, next_id,"negative demand")

                if "SDVRP" not in features and next_id in served_locations:
                    return _fail(vid, next_id, "repeated delivery requires SDVRP")
                served_locations.add(next_id)
                delivered[next_id] += quantity
                load_now += quantity
                if load_now > capacity:
                    return _fail(vid, next_id, "vehicle capacity exceeded")
            elif "MTVRP" in features and i + 1 < len(route) - 1:
                load_now = 0
            else:
                quantity = 0
            time += _service_time(next_loc, quantity, "SDVRP" in features)
            route_stops.append((next_id, quantity if next_id != 0 else 0, arrival_time, time))

        if time > vehicle["driver_tw"][1]:
            return _fail(vid, route[-1], "driver time window violated")

        total_time_consumed += time - vehicle["driver_tw"][0]
        driving_time += route_driving_time
        route_summaries.append((vid, route_stops, route_breaks))

    for location_id, location in locations.items():
        if location_id == 0:
            continue
        expected_demand = location.get("demand_sdvrp", location["demand"]) if "SDVRP" in features else location["demand"]
        if delivered[location_id] != expected_demand:
            return _fail(None, location_id, "total delivered quantity does not match location demand")

    if print_routes:
        print_route_summary(route_summaries)
    cost = driving_time + (10000 * len(used_vehicles) if "MTVRP" in features else 0)
    return {
        "feasible": True,
        "total_time_consumed": total_time_consumed,
        "driving_time": driving_time,
        "cost": cost,
    }
