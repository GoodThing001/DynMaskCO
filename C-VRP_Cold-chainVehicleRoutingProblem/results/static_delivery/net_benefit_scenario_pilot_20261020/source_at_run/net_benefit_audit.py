"""Independent raw-input recomputation; imports no solver/model cost function."""
import math


def audit(nodes, routes, c, rho, max_vehicles):
    errors, seen, tours = [], [], []
    if type(max_vehicles) is not int or max_vehicles <= 0 or len(routes) > max_vehicles:
        errors.append('fleet_bound')
    if not 0 <= rho <= 1:
        errors.append('rho_domain')
    tomato = c['tomato']
    fish = c['mullet_quality_counterfactual']
    temp = c['temperature']['product_and_cabin_C']
    heat = c['refrigeration_cost_scenario']
    heat_W = heat['conductivity_W_per_m_C'] * heat['surface_m2'] * (heat['ambient_C']-temp) / heat['wall_m']
    k_day = tomato['A_per_day'] * math.exp(-tomato['Ea_J_per_mol']/tomato['R_J_per_mol_K']/(temp+273.15))
    fish_rate = fish['k_ref_points_per_day'] * math.exp(fish['Ea_J_per_mol']/fish['R_J_per_mol_K'] * (1/(fish['T_ref_C']+273.15)-1/(temp+273.15)))
    revenue_parts, distance_parts, duration_parts = [], [], []
    min_r, min_fish = 1., fish['initial_score_approx']
    for route in routes:
        if not route or any(type(i) is not int or not 0 < i < len(nodes) for i in route):
            errors.append('invalid_route_customer')
            continue
        seen.extend(route)
        load = math.fsum(nodes[i][3] for i in route)
        if load > c['routing']['capacity_kg']+1e-7:
            errors.append('capacity')
        clock, start_clock, last = nodes[0][4], nodes[0][4], 0
        legs, trace, waiting = [], [], 0.
        for i in route:
            leg = math.hypot(nodes[i][1]-nodes[last][1], nodes[i][2]-nodes[last][2])
            legs.append(leg)
            arrival = clock + leg/c['routing']['speed_km_per_hour']*60
            service_start = max(arrival, nodes[i][4])
            if service_start > nodes[i][5]+1e-7:
                errors.append('time_window')
            waiting += service_start-arrival
            completion = service_start+nodes[i][6]
            days = (completion-start_clock)/1440
            retention = math.exp(-k_day*days)
            score = fish['initial_score_approx']-fish_rate*days
            price = tomato['fresh_price_EUR_per_kg']*(rho+(1-rho)*retention)
            income = nodes[i][3]*price
            revenue_parts.append(income)
            min_r, min_fish = min(min_r, retention), min(min_fish, score)
            trace.append({'customer': i, 'arrival_min': arrival, 'service_start_min': service_start, 'delivery_completion_min': completion, 'delivered_kg': nodes[i][3], 'tomato_TA_retention': retention, 'tomato_TA_g_per_kg_approx': tomato['initial_TA_g_per_kg_approx']*retention, 'hypothesized_price_EUR_per_kg': price, 'revenue_EUR': income, 'mullet_counterfactual_sensory_score_approx': score})
            last, clock = i, completion
        back = math.hypot(nodes[last][1]-nodes[0][1], nodes[last][2]-nodes[0][2])
        legs.append(back)
        clock += back/c['routing']['speed_km_per_hour']*60
        if clock > nodes[0][5]+1e-7:
            errors.append('depot_return')
        dist = math.fsum(legs)
        duration = clock-start_clock
        distance_parts.append(dist)
        duration_parts.append(duration)
        tours.append({'route': list(route), 'initial_load_kg': load, 'return_load_kg': 0, 'departure_min': start_clock, 'return_min': clock, 'waiting_min': waiting, 'distance_km': dist, 'duration_min': duration, 'trace': trace})
    if sorted(seen) != list(range(1, len(nodes))):
        errors.append('coverage_or_duplicate')
    mass = math.fsum(r[3] for r in nodes[1:])
    revenue = math.fsum(revenue_parts)
    purchase = mass*tomato['purchase_EUR_per_kg']
    distance = math.fsum(distance_parts)
    duration = math.fsum(duration_parts)
    traction = distance*c['routing']['distance_EUR_per_km']
    refrigeration = heat_W*duration*heat['coefficient_EUR_per_W_min']
    profit = revenue-purchase-traction-refrigeration
    nominal = mass*tomato['fresh_price_EUR_per_kg']
    return {'feasible_routing': not errors, 'errors': sorted(set(errors)), 'vehicles_used': len(routes), 'distance_km': distance, 'total_vehicle_minutes': duration, 'demand_kg': mass, 'nominal_revenue_EUR': nominal, 'sales_revenue_EUR': revenue, 'purchase_EUR': purchase, 'distance_cost_EUR': traction, 'refrigeration_cost_proxy_EUR': refrigeration, 'quality_discount_EUR': nominal-revenue, 'net_benefit_scenario_EUR': profit, 'normalized_net_benefit': profit/c['accounting']['currency_normalization_reference_EUR'], 'minimum_tomato_TA_retention': min_r, 'tomato_acceptance_threshold_configured': c['temperature']['tomato_min_TA_acceptance_threshold'] is not None, 'tomato_marketability_certified': False, 'minimum_mullet_counterfactual_sensory_score_approx': min_fish, 'mullet_counterfactual_sensory_threshold_passed': min_fish >= fish['minimum_sensory_score']-1e-8, 'mullet_profit_computed': False, 'thermal_wall_heat_leak_W': heat_W, 'electric_input_power_claimed': False, 'rho': rho, 'tours': tours}
