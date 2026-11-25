import os
import json
import osmnx as ox
import folium
import numpy as np
from shapely.geometry import Polygon
from scipy.stats import truncnorm

from config import *


# ------------------------------------------------------------------
# --- Initialize folders ---
# ------------------------------------------------------------------
def initialize_folders(cities):
    for city in cities:
        os.makedirs(os.path.join(DATA_PATH, city), exist_ok=True)
        os.makedirs(os.path.join(DATA_PATH, city, 'tmp'), exist_ok=True)
        os.makedirs(os.path.join(DATA_PATH, city, 'tests'), exist_ok=True)
        os.makedirs(os.path.join(DATA_PATH, city, 'maps'), exist_ok=True)
        os.makedirs(os.path.join(DATA_PATH, city, 'solutions'), exist_ok=True)


# ------------------------------------------------------------------
# --- City selection ---
# ------------------------------------------------------------------

def select_input_cities(cities_to_select=None, parameters_to_return=None):
    # --- Define cities ---
    cities = {
    # S instances: 50-100
        "Genoa": {           
            "region": (8.72, 44.35, 9.10, 44.50),
            "depot": (44.406, 8.899),  # Porto di Genova, center
            "n_supermarkets": 50,
            "n_convenience": 25 ,
            "warehouse_tw": ("05:30", "13:00"),
            "driver_tw": ("05:30", "13:00"),
            "break_mins": 45,
            "driving_before_break_mins": 270,
            "supermarket_delivery_tw": ("06:00", "12:00"),
            "convenience_delivery_tw": ("06:00", "11:00"),
        },
        "Lyon": {           
            "region": (4.75, 45.69, 4.98, 45.85),
            "depot": (45.727, 4.840),  # Gerland industrial area
            "n_supermarkets": 0,
            "n_convenience": 85,
            "warehouse_tw": ("05:30", "12:30"),
            "driver_tw": ("05:30", "12:30"),
            "break_mins": 45,
            "driving_before_break_mins": 270,
            "supermarket_delivery_tw": ("06:00", "11:30"),
            "convenience_delivery_tw": ("06:00", "11:30"),
        },
        "Greater Oklahoma City": {           
            "region": (-97.70, 35.16, -97.41, 35.73),
            "depot": (35.437, -97.653),  # OKC Logistics park
            "n_supermarkets": 20,
            "n_convenience": 40,
            "warehouse_tw": ("05:30", "19:00"),
            "driver_tw": ("05:30", "19:00"),
            "break_mins": 30,
            "driving_before_break_mins": 480,
            "supermarket_delivery_tw": ("06:00", "15:00"),
            "convenience_delivery_tw": ("06:00", "18:00"),
        },
        "Central Dalmatia": {           
            "region": [(16.05, 43.24), (16.05, 43.73), (16.55, 43.73), (16.87, 43.68), (17.09, 43.24), (16.55, 43.24)],
            "depot": (43.592, 16.579),  # Dugopolje industrial zone
            "n_supermarkets": 50,
            "n_convenience": 0,
            "warehouse_tw": ("05:00", "11:00"),
            "driver_tw": ("05:00", "11:00"),
            "break_mins": 45,
            "driving_before_break_mins": 270,
            "supermarket_delivery_tw": ("05:30", "10:00"),
            "convenience_delivery_tw": ("05:30", "10:00"),
        },

        # M instances: 100-200
        "Zagreb": {           
            "region": (15.84, 45.88, 16.14, 45.73),
            "depot": (45.800, 15.878),  # Jankomir logistics & warehouse zone
            "n_supermarkets": 0,
            "n_convenience": 170,
            "warehouse_tw": ("05:30", "13:00"),
            "driver_tw": ("05:30", "13:00"),
            "break_mins": 45,
            "driving_before_break_mins": 270,
            "supermarket_delivery_tw": ("06:00", "12:00"),
            "convenience_delivery_tw": ("06:00", "12:00"),
        },
        "Brasilia": {           
            "region": (-48.21, -15.96, -47.73, -15.62),  # No south part
            "depot": (-15.793, -47.971),  # Cidade do Automóvel - main logistics area
            "n_supermarkets": 100,
            "n_convenience": 0,
            "warehouse_tw": ("05:30", "15:00"),
            "driver_tw": ("05:30", "15:00"),
            "break_mins": 30,
            "driving_before_break_mins": 330,
            "supermarket_delivery_tw": ("06:00", "14:00"),
            "convenience_delivery_tw": ("06:00", "14:00"),
        },
        "Greater Copenhagen": {           
            "region": (12.30, 55.56, 13.26, 55.81),
            "depot": (55.657, 12.420),  # Brøndby industrial area
            "n_supermarkets": 100,
            "n_convenience": 50,
            "warehouse_tw": ("05:00", "12:00"),
            "driver_tw": ("05:00", "12:00"),
            "break_mins": 45,
            "driving_before_break_mins": 270,
            "supermarket_delivery_tw": ("05:30", "11:00"),
            "convenience_delivery_tw": ("05:30", "10:30"),
        },
        "Ruhr Area": {           
            "region": (6.69, 51.32, 7.60, 51.56),
            "depot": (51.439, 7.092),  # Duisburg Logport
            "n_supermarkets": 40,
            "n_convenience": 80,
            "warehouse_tw": ("04:30", "15:00"),
            "driver_tw": ("04:30", "15:00"),
            "break_mins": 45,
            "driving_before_break_mins": 270,
            "supermarket_delivery_tw": ("05:00", "12:30"),
            "convenience_delivery_tw": ("05:00", "14:00"),
        },

        # L instances: 200-400
        "Singapore": {        
            "region": [(103.53, 1.19), (103.68, 1.42), (103.82, 1.475), (104.01, 1.39), (104.09, 1.29), (103.74, 1.15)],
            "depot": (1.319, 103.869),  # Kallang / Central-East logistics
            "n_supermarkets": 200,
            "n_convenience": 0,
            "warehouse_tw": ("05:30", "14:30"),
            "driver_tw": ("05:30", "14:30"),
            "break_mins": 60,
            "driving_before_break_mins": 300,
            "supermarket_delivery_tw": ("06:00", "13:30"),
            "convenience_delivery_tw": ("06:00", "13:30"),
        },
        "Melbourne": {           
            "region": (144.75, -38.40, 145.25, -37.60),
            "depot": (-37.835, 144.740),  # Laverton North / Truganina logistics hub
            "n_supermarkets": 113,
            "n_convenience": 227,
            "warehouse_tw": ("05:30", "18:00"),
            "driver_tw": ("05:30", "18:00"),
            "break_mins": 60,
            "driving_before_break_mins": 510,  # Real constraint is more complex
            "supermarket_delivery_tw": ("06:00", "15:00"),
            "convenience_delivery_tw": ("06:00", "17:00"),
        },
        "Greater Cairo": {           
            "region": (30.86, 29.70, 31.83, 30.53),  # Vjerojatno previše stanovnika za L, možda odsijeći sjever
            "depot": (29.993, 31.476),  # New Cairo Industrial Area
            "n_supermarkets": 160,
            "n_convenience": 80,
            "warehouse_tw": ("05:30", "17:00"),
            "driver_tw": ("05:30", "17:00"),
            "break_mins": 60,
            "driving_before_break_mins": 300,
            "supermarket_delivery_tw": ("06:00", "16:00"),
            "convenience_delivery_tw": ("06:00", "15:30"),
        },
        "SF Bay Area": {           
            "region": [(-122.61, 38.18), (-122.57, 37.48), (-122.00, 37.17), (-121.47, 37.29), (-121.70, 37.70), (-122.20, 38.22)],
            "depot": (37.650, -122.397),  # East of 101 industrial and logistics area (South San Francisco)
            "n_supermarkets": 0,
            "n_convenience": 300,
            "warehouse_tw": ("05:30", "17:30"),
            "driver_tw": ("05:30", "17:30"),
            "break_mins": 30,
            "driving_before_break_mins": 480,
            "supermarket_delivery_tw": ("06:00", "16:30"),
            "convenience_delivery_tw": ("06:00", "16:30"),
        },

        # XL instances: 400-1000
        "London": {           
            "region": (-0.43, 51.32, 0.20, 51.66),
            "depot": (51.523, 0.142),  # Thames Gateway industrial zone
            "n_supermarkets": 0,
            "n_convenience": 1000,
            "warehouse_tw": ("06:00", "14:00"),
            "driver_tw": ("06:00", "14:00"),
            "break_mins": 45,
            "driving_before_break_mins": 270,
            "supermarket_delivery_tw": ("06:30", "13:00"),
            "convenience_delivery_tw": ("06:30", "13:00"),
        },
        "Istanbul": {           
            "region": (28.55, 40.85, 29.50, 41.20),
            "depot": (41.085, 28.793),  # Ikitelli Organized Industrial Zone
            "n_supermarkets": 400,
            "n_convenience": 0,
            "warehouse_tw": ("04:30", "14:00"),
            "driver_tw": ("04:30", "14:00"),
            "break_mins": 45,
            "driving_before_break_mins": 270,
            "supermarket_delivery_tw": ("05:00", "13:00"),
            "convenience_delivery_tw": ("05:00", "13:00"),
        },
        "Mexico City Metro": {
            "region": [(-99.40, 19.20), (-99.20, 19.05), (-98.80, 19.15), (-98.70, 19.50), (-99.00, 19.70), (-99.40, 19.55)],
            "depot": (19.483, -99.162),  # Vallejo Industrial Area (Azcapotzalco)
            "n_supermarkets": 267,
            "n_convenience": 533,
            "warehouse_tw": ("04:30", "15:00"), #("00:00", "23:59"),
            "driver_tw": ("04:30", "15:00"),
            "break_mins": 30,
            "driving_before_break_mins": 300,
            "supermarket_delivery_tw": ("05:00", "14:00"),
            "convenience_delivery_tw": ("05:00", "13:30"),
        },
        "Greater Bay Area": {           
            "region": [(112.95, 23.45), (113.00, 22.10), (113.80, 21.40), (114.70, 21.60), (114.10, 23.00), (113.80, 23.45)],  # No Zhaoqing
            "depot": (22.500, 113.883),  # Shenzhen Qianhai Bay
            "n_supermarkets": 400,
            "n_convenience": 200,
            "warehouse_tw": ("05:30", "19:00"), #("00:00", "23:59"),
            "driver_tw": ("05:30", "19:00"),
            "break_mins": 20,
            "driving_before_break_mins": 240,
            "supermarket_delivery_tw": ("06:00", "15:30"),
            "convenience_delivery_tw": ("06:00", "18:00"),
        }
    }
    
    # --- Filter cities ---
    if cities_to_select:
        cities = {c: d for c, d in cities.items() if c in cities_to_select}

    # --- Filter parameters ---
    if parameters_to_return:
        for city in cities:
            cities[city] = {k: v for k, v in cities[city].items() if k in parameters_to_return}

    return cities
    

# ------------------------------------------------------------------
# --- Main location generator ---
# ------------------------------------------------------------------

def generate_locations_osm(cities, generate_map=True):
    os.makedirs(DATA_PATH, exist_ok=True)

    for city, cfg in cities.items():
        print(f"\n=== {city} ===")
        city_dir = os.path.join(DATA_PATH, city)
        os.makedirs(city_dir, exist_ok=True)

        wh_tw = _hhmm_to_sec_tuple(cfg["warehouse_tw"])
        sm_tw = _hhmm_to_sec_tuple(cfg["supermarket_delivery_tw"])
        cv_tw = _hhmm_to_sec_tuple(cfg["convenience_delivery_tw"])

        # --- Fetch POIs ---
        gdf_sm, gdf_cv, center_lat, center_lon = _fetch_pois(cfg)
        sm_sample = gdf_sm.sample(n=min(cfg["n_supermarkets"], len(gdf_sm)), random_state=99)
        cv_sample = gdf_cv.sample(n=min(cfg["n_convenience"], len(gdf_cv)), random_state=99)

        # --- Generate demands ---
        seed = cfg['n_supermarkets'] + cfg['n_convenience']
        sm_demands = _generate_demands(len(sm_sample), 6, 0.6, 2.5, 14, 0.5, seed)
        cv_demands = _generate_demands(len(cv_sample), 1.5, 0.6, 0.5, 3.5, 0.5, seed)

        # --- Build dataset ---
        dataset = _build_dataset(cfg, sm_sample, cv_sample, sm_demands, cv_demands, wh_tw, sm_tw, cv_tw)
        dataset = _add_sdvrp_demands(dataset, seed=seed)

        # --- Save JSON ---
        out_json = os.path.join(city_dir, FILES["locations"])
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved dataset: {out_json}")

        # --- Optional map ---
        if generate_map:
            _save_map(city, city_dir, dataset, center_lat, center_lon)


def generate_all_locations_osm(cities, generate_map=True):
    os.makedirs(DATA_PATH, exist_ok=True)

    for city, cfg in cities.items():
        print(f"\n=== {city} ===")
        city_dir = os.path.join(DATA_PATH, city)
        os.makedirs(city_dir, exist_ok=True)

        wh_tw = _hhmm_to_sec_tuple(cfg["warehouse_tw"])
        sm_tw = _hhmm_to_sec_tuple(cfg["supermarket_delivery_tw"])
        cv_tw = _hhmm_to_sec_tuple(cfg["convenience_delivery_tw"])

        # --- Fetch POIs ---
        sm_sample, cv_sample, center_lat, center_lon = _fetch_pois(cfg)

        # --- Generate demands ---
        seed = cfg['n_supermarkets'] + cfg['n_convenience']
        sm_demands = _generate_demands(len(sm_sample), 6, 0.6, 2.5, 14, 0.5, seed)
        cv_demands = _generate_demands(len(cv_sample), 1.5, 0.6, 0.5, 3.5, 0.5, seed)

        # --- Build dataset ---
        dataset = _build_dataset(cfg, sm_sample, cv_sample, sm_demands, cv_demands, wh_tw, sm_tw, cv_tw)
        dataset = _add_sdvrp_demands(dataset, seed=seed)

        # --- Save JSON ---
        out_json = os.path.join(city_dir, FILES["all_available_locations"])
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"Saved dataset: {out_json}")

        # --- Optional map ---
        if generate_map:
            _save_map(city, city_dir, dataset, center_lat, center_lon, map_filename="all_available_locations_map.html")


def fix_unfeasible_solutions(data_path=DATA_PATH):
    replacements = []

    for city in sorted(os.listdir(data_path)):
        city_path = os.path.join(data_path, city)
        if not os.path.isdir(city_path):
            continue

        locations_path = os.path.join(city_path, FILES["locations"])
        accessibility_path = os.path.join(city_path, FILES["vehicle_location_accessibility"])
        if not os.path.isfile(locations_path) or not os.path.isfile(accessibility_path):
            print(f"{city}: locations or accessibility JSON not found; skipping.")
            continue

        with open(locations_path, "r", encoding="utf-8") as locations_file:
            locations = json.load(locations_file)
        with open(accessibility_path, "r", encoding="utf-8") as accessibility_file:
            accessibility = json.load(accessibility_file)

        accessibility_matrix = accessibility["accessibility_matrix"]
        if len(accessibility_matrix) <= 2:
            print(f"{city}: no third row in accessibility_matrix; skipping.")
            continue

        vehicle_accessibility = accessibility_matrix[2]
        if len(vehicle_accessibility) < len(locations):
            raise ValueError(
                f"{city}: third row accessibility_matrix has "
                f"{len(vehicle_accessibility)} columns for {len(locations)} locations."
            )

        problematic_indices = [index for index, location in enumerate(locations)
                               if location.get("demand_sdvrp", 0) > 8 and vehicle_accessibility[index] == 0]
        if not problematic_indices:
            print(f"{city}: no problematic locations.")
            continue

        print(f"{city}: problematic IDs: {[locations[index]['id'] for index in problematic_indices]}")

        city_replacements = []
        for location_index in problematic_indices:
            candidate_indices = list(range(location_index + 1, len(locations))) + list(range(location_index - 1, -1, -1))

            replacement_index = next(
                (index for index in candidate_indices
                 if locations[index].get("location_type") != "depot" and locations[index].get("demand_sdvrp") is not None
                 and locations[index]["demand_sdvrp"] <= 8 and vehicle_accessibility[index] != 0),
                None)
            if replacement_index is None:
                print(f"{city}: replacement location not found for ID {locations[location_index]['id']}.")
                continue

            problem_location = locations[location_index]
            replacement_location = locations[replacement_index]
            problem_id = problem_location["id"]
            replacement_id = replacement_location["id"]
            for key in problem_location:
                if key in {"id", "x_coor", "y_coor"}:
                    continue
                problem_location[key], replacement_location[key] = (
                    replacement_location[key],
                    problem_location[key],
                )

            city_replacements.append((problem_id, replacement_id))
            replacements.append((city, problem_id, replacement_id))
            print(f"{city}: ID {problem_id} was replaced with ID {replacement_id}.")

        if city_replacements:
            with open(locations_path, "w", encoding="utf-8") as locations_file:
                json.dump(locations, locations_file, indent=2, ensure_ascii=False)
                locations_file.write("\n")

    return replacements



# ------------------------------------------------------------------
# --- Helper functions ---
# ------------------------------------------------------------------

def _hhmm_to_sec_tuple(tw):
    h1, m1 = map(int, tw[0].split(":"))
    h2, m2 = map(int, tw[1].split(":"))
    return (h1 * 3600 + m1 * 60, h2 * 3600 + m2 * 60)


def _fetch_pois(cfg):
    tags_sm = {"shop": ["supermarket"]}
    tags_cv = {"shop": ["convenience"]}
    region = cfg["region"]

    if isinstance(region, (list, tuple)) and len(region) == 4:
        west, south, east, north = region
        polygon = None
        center_lat = (south + north) / 2
        center_lon = (west + east) / 2
    else:
        polygon = Polygon(region)
        lats = [p[1] for p in region]
        lons = [p[0] for p in region]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

    gdf_sm = ox.features_from_polygon(polygon, tags_sm) if polygon else ox.features_from_bbox((west, south, east, north), tags_sm)
    gdf_cv = ox.features_from_polygon(polygon, tags_cv) if polygon else ox.features_from_bbox((west, south, east, north), tags_cv)
    gdf_sm = gdf_sm[gdf_sm.geometry.type == "Point"].to_crs("EPSG:4326")
    gdf_cv = gdf_cv[gdf_cv.geometry.type == "Point"].to_crs("EPSG:4326")

    return gdf_sm, gdf_cv, center_lat, center_lon


def _build_dataset(cfg, sm_sample, cv_sample, sm_demands, cv_demands, wh_tw, sm_tw, cv_tw):
    dataset = []

    # --- Depot ---
    depot_lat, depot_lon = cfg["depot"]
    dataset.append({
        "id": 0,
        "x_coor": depot_lon,
        "y_coor": depot_lat,
        "demand": 0,
        "location_type": "depot",
        "time_window": list(wh_tw),
        "service_time": 30 * 60
    })

    # --- Supermarkets ---
    for (_, row), demand in zip(sm_sample.iterrows(), sm_demands):
        dataset.append({
            "id": len(dataset),
            "x_coor": row.geometry.x,
            "y_coor": row.geometry.y,
            "demand": demand,
            "location_type": "supermarket",
            "time_window": list(sm_tw),
            "service_time": int(5 * 60 + demand * 120)
        })

    # --- Convenience ---
    for (_, row), demand in zip(cv_sample.iterrows(), cv_demands):
        dataset.append({
            "id": len(dataset),
            "x_coor": row.geometry.x,
            "y_coor": row.geometry.y,
            "demand": demand,
            "location_type": "convenience",
            "time_window": list(cv_tw),
            "service_time": int(5 * 60 + demand * 90)
        })

    return dataset


def _add_sdvrp_demands(dataset, seed=100):
    """Creates SDVRP demands while preserving the original demand distribution for the large majority of customers.
    Demand modifications:
        - 90% remain unchanged.
        - 7% are selected from the highest-demand 50% and multiplied
          by a random factor in [1.2, 2.0].
        - 3% are selected from the lowest-demand 50% and receive a
          fixed additive increase:
              * supermarket: +26
              * convenience: +18
    The targeted 3% are selected globally, with preference for supermarkets. At least one targeted customer is guaranteed if the dataset contains an eligible customer.
    All SDVRP demands are rounded to the nearest 0.5."""

    rng = np.random.default_rng(seed)
    n = len(dataset)
    if n == 0:
        return dataset

    # Initialize SDVRP fields with original values
    for index, point in enumerate(dataset):
        ordered_point = {}
        for key, value in point.items():
            ordered_point[key] = value
            if key == "demand":
                ordered_point["demand_sdvrp"] = value
            if key == "service_time":
                ordered_point["service_time_sdvrp"] = value
        dataset[index] = ordered_point

    # Number of customers in each modification group
    n_multiplier = int(np.ceil(n * 0.07))
    n_targeted = int(np.ceil(n * 0.03))
    n_multiplier = min(n_multiplier, n)
    n_targeted = min(n_targeted, n - n_multiplier)

    # Build top-50% and bottom-50% pools
    sorted_indices = sorted(range(n), key=lambda i: dataset[i]["demand"], reverse=True)
    n_top = max(1, int(np.ceil(n * 0.50)))
    top_pool = sorted_indices[:n_top]
    bottom_pool = sorted_indices[-n_top:]

    # Select 7% for multiplicative increase
    multiplier_indices = rng.choice(top_pool, size=min(n_multiplier, len(top_pool)), replace=False)
    multiplier_indices = set(multiplier_indices)
    for index in multiplier_indices:
        original_demand = dataset[index]["demand"]
        multiplier = rng.uniform(1.2, 2.0)
        demand_sdvrp = (round(original_demand * multiplier / 0.5) * 0.5)
        dataset[index]["demand_sdvrp"] = demand_sdvrp

    # Select 3% targeted customers from bottom 50%, prefer supermarkets.
    available_bottom = [i for i in bottom_pool if i not in multiplier_indices]
    supermarket_candidates = [i for i in available_bottom if dataset[i]["location_type"] == "supermarket"]
    convenience_candidates = [i for i in available_bottom if dataset[i]["location_type"] == "convenience"]

    # Prefer supermarkets. If there are not enough supermarkets, fill the remaining positions with convenience customers.
    n_targeted_supermarkets = min(n_targeted, len(supermarket_candidates))
    selected_targeted = []

    if n_targeted_supermarkets > 0:
        selected_targeted.extend(rng.choice(supermarket_candidates, size=n_targeted_supermarkets, replace=False))
    remaining = n_targeted - len(selected_targeted)
    if remaining > 0:
        remaining_convenience = [i for i in convenience_candidates if i not in selected_targeted]
        selected_targeted.extend(rng.choice(remaining_convenience, size=remaining, replace=False))

    # Apply targeted additive increases
    for index in selected_targeted:
        original_demand = dataset[index]["demand"]
        if dataset[index]["location_type"] == "supermarket":
            increment = 26
        else:
            increment = 18
        demand_sdvrp = original_demand + increment
        demand_sdvrp = (round(demand_sdvrp / 0.5) * 0.5)
        dataset[index]["demand_sdvrp"] = demand_sdvrp

    # Recalculate SDVRP service times
    for point in dataset:
        demand_sdvrp = point["demand_sdvrp"]
        if point["location_type"] == "supermarket":
            point["service_time_sdvrp"] = int(5 * 60 + demand_sdvrp * 120)
        elif point["location_type"] == "convenience":
            point["service_time_sdvrp"] = int(5 * 60 + demand_sdvrp * 90)

    return dataset


def _save_map(city, city_dir, dataset, center_lat, center_lon, map_filename="locations_map.html"):
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="cartodbpositron"
    )

    for p in dataset:
        tooltip_text = (
            f"ID: {p['id']} | "
            f"Type: {p['location_type']} | "
            f"Demand: {p['demand']} | "
            f"TW: {p['time_window']} | "
            f"Service: {p['service_time']} s"
        )
        if p["location_type"] == "depot":
            folium.Marker(
                location=[p["y_coor"], p["x_coor"]],
                icon=folium.Icon(color="red", icon="home"),
                tooltip=tooltip_text
            ).add_to(m)
        else:
            color = "green" if p["location_type"] == "supermarket" else "orange"
            folium.CircleMarker(
                location=[p["y_coor"], p["x_coor"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.85,
                tooltip=tooltip_text
            ).add_to(m)

    out_map = os.path.join(city_dir, "maps", map_filename)
    os.makedirs(os.path.dirname(out_map), exist_ok=True)
    m.save(out_map)
    print(f"Saved locations map: {out_map}")


# ------------------------------------------------------------------
# --- Demands generator ---
# ------------------------------------------------------------------

def _generate_demands(n_points=100_000, mean=8.0, sigma=0.5, min_demand=0.5, max_demand=16.0, step=0.5, seed=100):
    np.random.seed(seed)
    mu = np.log(mean) - 0.5 * sigma**2
    a, b = (np.log(min_demand) - mu) / sigma, (np.log(max_demand) - mu) / sigma
    d = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_points)
    return np.round(np.exp(d) / step) * step


# ------------------------------------------------------------------
# --- Vehicles generator ---
# ------------------------------------------------------------------

def generate_vehicles(cities, capacity=18, buffer=1.4):
    for city, cfg in cities.items():
        dataset_path = os.path.join(DATA_PATH, city, FILES["locations"])
        out_json = os.path.join(DATA_PATH, city, FILES["vehicles"])
        total_demand = _compute_total_sdvrp_demand(dataset_path)
        driver_tw = _hhmm_to_sec_tuple(cfg["driver_tw"])

        # --- Vehicle pattern ---
        large_dominant = city not in ["Lyon", "Oklahoma City", "Zagreb", "Melbourne", "London", "Tokyo"]
        pattern = (
            [8]*1 + [18]*2 + [26]*2     # Big vehicles dominant
            if large_dominant else
            [8]*2 + [18]*2 + [26]*1     # Small vehicles dominant
        )

        # --- Build fleet ---
        vehicles = []
        total_capacity = 0
        vehicle_id = 0

        while total_capacity < total_demand * buffer:
            for hvrp_capacity in pattern:
                vehicles.append({
                    "id": vehicle_id,
                    "capacity": capacity,
                    "hvrp_capacity": hvrp_capacity,
                    "driver_tw": list(driver_tw),
                    "break_mins": cfg["break_mins"],
                    "driving_before_break_mins": cfg["driving_before_break_mins"]
                })
                total_capacity += hvrp_capacity
                vehicle_id += 1
                if total_capacity >= total_demand * buffer:
                    break

        # --- Save JSON ---
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(vehicles, f, indent=2)
        print(f"Saved vehicles to: {out_json}")



def _compute_total_sdvrp_demand(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return sum(p["demand_sdvrp"] for p in data if p["location_type"] != "depot")