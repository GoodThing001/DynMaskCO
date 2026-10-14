"""Independent depot-to-customer contract; C0 is a read-only physics reference."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'coldchain'))
from coldchain_contract import ColdChainContractV2, UnitScale


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'not_applicable'
    return value


@dataclass(frozen=True)
class DeliveryContract:
    version: str = 'delivery-coldchain-engineering-v1'
    mode: str = 'depot_to_customer'
    capacity: float = 50.0
    vehicles: int = 5
    horizon_h: float = 8.0
    speed_kmph: float = 30.0
    loading_service_h: float = 0.05
    wave_interval_h: float = 0.5
    distance_weight: float = 1.0
    energy_weight: float = 0.1
    quality_weight: float = 1.0
    product_value: tuple = (1.0, 1.5, 2.0)
    quality_clock: str = 'loading_finish_to_customer_service_finish'
    temperature_hard_scope: str = 'occupied_compartments'
    storage_scope: str = 'external_control_not_policy_cost'
    service_time_window: str = 'customer_service_start'
    cargo_rule: str = 'all_route_orders_loaded_before_departure'
    stage: str = 'pre_dispatch_route_snapshot_engineering'
    online_ready: bool = False

    def physics(self):
        # Only units, thermal parameters and quality kinetics are reused.
        # No pickup transition, pickup manifest, or depot-unload event is called.
        return ColdChainContractV2(units=UnitScale(1.0, 1.0, self.speed_kmph))

    def validate(self):
        if self.mode != 'depot_to_customer' or self.version != 'delivery-coldchain-engineering-v1':
            raise ValueError('Unsupported delivery contract')
        if self.online_ready or self.stage != 'pre_dispatch_route_snapshot_engineering':
            raise ValueError('Online dispatch requires a separately implemented and registered contract')
        nums = (self.capacity, self.horizon_h, self.speed_kmph, self.loading_service_h,
                self.wave_interval_h, self.distance_weight, self.energy_weight, self.quality_weight)
        if any(not math.isfinite(v) or v < 0 for v in nums):
            raise ValueError('Nonfinite or negative delivery parameter')
        if min(self.capacity, self.horizon_h, self.speed_kmph, self.wave_interval_h) <= 0:
            raise ValueError('Invalid capacity/horizon/speed/waves')
        if type(self.vehicles) is not int or self.vehicles < 1:
            raise ValueError('Invalid finite fleet')
        expected = {'quality_clock': 'loading_finish_to_customer_service_finish',
            'temperature_hard_scope': 'occupied_compartments',
            'storage_scope': 'external_control_not_policy_cost',
            'service_time_window': 'customer_service_start',
            'cargo_rule': 'all_route_orders_loaded_before_departure'}
        if any(getattr(self, k) != v for k, v in expected.items()):
            raise ValueError('Undeclared delivery semantics change')
        if len(self.product_value) != 3 or any(not math.isfinite(v) or v <= 0 for v in self.product_value):
            raise ValueError('Invalid product values')
        self.physics().validate()

    def payload(self):
        self.validate()
        p = self.physics()
        return {'delivery': asdict(self), 'physics': finite_json({'units': asdict(p.units),
            'thermal': asdict(p.thermal), 'quality': asdict(p.quality)}),
            'parameter_scope': 'pilot_reference_not_real_vehicle_calibration',
            'quality_note': 'v2 fixed rates do not model a new temperature-dependent kinetic law',
            'objective_note': 'engineering scalarization; formal pilot must preregister units/weights',
            'excluded': ['charging', 'in_transit_reloading', 'preemptive_return', 'stock_shortage']}

    def identity(self):
        raw = json.dumps(self.payload(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()
