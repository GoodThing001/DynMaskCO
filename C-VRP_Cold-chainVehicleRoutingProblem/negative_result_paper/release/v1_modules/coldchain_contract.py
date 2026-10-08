"""Versioned physical and objective contract for pickup-to-depot cold-chain routing.

This module contains configuration and unit conversion only.  The transition
implementation lives in ``coldchain_state.py`` so every consumer can share one
set of physics and accounting rules.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any


SCHEMA_VERSION = "coldchain-contract-v1"
_ALLOWED_SOURCE_TYPES = frozenset({"literature", "calibrated", "pilot"})


@dataclass(frozen=True)
class OperationalConfig:
    mode: str = "pickup_to_depot"
    fleet_model: str = "homogeneous_multi_compartment"
    capacity_model: str = "shared_total"
    shared_vehicle_capacity: float = 50.0
    single_trip: bool = True
    allow_reload: bool = False
    quality_clock: str = "service_finish_to_return_arrival"


@dataclass(frozen=True)
class UnitScale:
    """Mapping from dataset units to physical units."""

    distance_km_per_unit: float = 1.0
    hours_per_time_unit: float = 1.0
    speed_kmph: float = 1.0


@dataclass(frozen=True)
class ThermalConfig:
    """Three-zone thermal parameters ordered as ambient/chilled/frozen.

    ``cooling_power_kw`` is electrical input power.  Temperature dynamics use
    the separately declared ``cooling_rate_c_per_hour`` pilot coefficient;
    this avoids silently treating kW as degrees Celsius.
    """

    ambient_temperature_c: float = 25.0
    target_temperature_c: tuple[float, ...] = (18.0, 4.0, -18.0)
    hard_temperature_bounds_c: tuple[tuple[float, float], ...] = (
        (0.0, 30.0),
        (0.0, 8.0),
        (-25.0, -12.0),
    )
    heat_transfer_per_hour: tuple[float, ...] = (0.20, 0.16, 0.12)
    cooling_power_kw: tuple[float, ...] = (0.30, 1.80, 3.20)
    cooling_rate_c_per_hour: tuple[float, ...] = (2.0, 5.0, 8.0)
    cop_parameters: tuple[float, float, float] = (3.0, 0.03, 1.2)
    door_heat_kwh: tuple[float, ...] = (0.010, 0.035, 0.060)
    thermal_capacity_kwh_per_c: tuple[float, ...] = (0.30, 0.40, 0.50)
    dispatch_preconditioning_energy_kwh: tuple[float, ...] = (0.05, 0.15, 0.30)
    supported_temp_classes: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class QualityConfig:
    """Arrhenius quality-decay parameters ordered by temperature class."""

    reference_rate_per_hour: tuple[float, ...] = (0.0010, 0.0020, 0.0035)
    activation_over_r_kelvin: tuple[float, ...] = (3500.0, 4500.0, 5500.0)
    reference_temperature_k: float = 277.15
    unsalable_quality_threshold: tuple[float, ...] = (0.80, 0.85, 0.90)
    product_value: tuple[float, ...] = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class ColdChainObjectiveConfig:
    distance_scale: float = 1.0
    quality_scale: float = 1.0
    energy_scale: float = 1.0
    lambda_quality: float = 1.0
    lambda_energy: float = 0.1


@dataclass(frozen=True)
class ParameterProvenance:
    """Source and preregistered sensitivity interval for one parameter family."""

    parameter_path: str
    source_type: str
    source_reference: str
    sensitivity_low: float
    sensitivity_high: float
    unit: str


def _default_provenance() -> tuple[ParameterProvenance, ...]:
    return (
        ParameterProvenance(
            "units.distance_km_per_unit", "pilot", "C0 pilot contract",
            0.1, 10.0, "km / dataset_distance_unit"),
        ParameterProvenance(
            "units.hours_per_time_unit", "pilot", "C0 pilot contract",
            0.25, 2.0, "h / dataset_time_unit"),
        ParameterProvenance(
            "units.speed_kmph", "pilot", "C0 pilot contract",
            0.1, 100.0, "km / h"),
        ParameterProvenance(
            "thermal.ambient_temperature_c", "pilot", "C0 pilot contract",
            20.0, 40.0, "degC"),
        ParameterProvenance(
            "thermal.heat_transfer_per_hour", "pilot", "C0 pilot contract",
            0.05, 0.40, "1 / h"),
        ParameterProvenance(
            "thermal.target_temperature_c", "pilot", "C0 pilot contract",
            -25.0, 25.0, "degC"),
        ParameterProvenance(
            "thermal.hard_temperature_bounds_c", "pilot", "C0 pilot contract",
            -35.0, 45.0, "degC"),
        ParameterProvenance(
            "thermal.cooling_power_kw", "pilot", "C0 pilot contract",
            0.0, 6.0, "kW_electric"),
        ParameterProvenance(
            "thermal.cooling_rate_c_per_hour", "pilot", "C0 pilot contract",
            0.0, 12.0, "degC / h"),
        ParameterProvenance(
            "thermal.cop_parameters", "pilot", "C0 pilot contract",
            0.0, 6.0, "mixed COP pilot coefficients"),
        ParameterProvenance(
            "thermal.door_heat_kwh", "pilot", "C0 pilot contract",
            0.0, 0.15, "kWh_thermal / opening"),
        ParameterProvenance(
            "thermal.thermal_capacity_kwh_per_c", "pilot", "C0 pilot contract",
            0.1, 1.0, "kWh_thermal / degC"),
        ParameterProvenance(
            "thermal.dispatch_preconditioning_energy_kwh", "pilot",
            "C0 pilot contract", 0.0, 1.0, "kWh / dispatch"),
        ParameterProvenance(
            "quality.reference_rate_per_hour", "pilot", "C0 pilot contract",
            0.0001, 0.02, "1 / h"),
        ParameterProvenance(
            "quality.activation_over_r_kelvin", "pilot", "C0 pilot contract",
            1000.0, 10000.0, "K"),
        ParameterProvenance(
            "quality.reference_temperature_k", "pilot", "C0 pilot contract",
            260.0, 310.0, "K"),
        ParameterProvenance(
            "quality.unsalable_quality_threshold", "pilot", "C0 pilot contract",
            0.5, 1.0, "quality fraction"),
        ParameterProvenance(
            "quality.product_value", "pilot", "C0 pilot contract",
            0.5, 3.0, "relative value / quantity"),
        ParameterProvenance(
            "objective.distance_scale", "pilot", "C0 pilot contract (devmean-normalized)",
            0.001, 100000.0, "distance normalization"),
        ParameterProvenance(
            "objective.quality_scale", "pilot", "C0 pilot contract (devmean-normalized)",
            0.001, 100000.0, "quality normalization"),
        ParameterProvenance(
            "objective.energy_scale", "pilot", "C0 pilot contract (devmean-normalized)",
            0.001, 100000.0, "energy normalization"),
        ParameterProvenance(
            "objective.lambda_quality", "pilot", "C0 pilot contract",
            0.0, 5.0, "dimensionless"),
        ParameterProvenance(
            "objective.lambda_energy", "pilot", "C0 pilot contract",
            0.0, 5.0, "dimensionless"),
    )


@dataclass(frozen=True)
class ColdChainContract:
    schema_version: str = SCHEMA_VERSION
    operational: OperationalConfig = field(default_factory=OperationalConfig)
    units: UnitScale = field(default_factory=UnitScale)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    objective: ColdChainObjectiveConfig = field(default_factory=ColdChainObjectiveConfig)
    parameter_provenance: tuple[ParameterProvenance, ...] = field(
        default_factory=_default_provenance)

    def validate(self) -> None:
        """Reject ambiguous units, mutable semantics, and untagged pilot values."""

        op = self.operational
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported cold-chain schema_version: %s" % self.schema_version)
        if op.mode != "pickup_to_depot":
            raise ValueError("C0 requires pickup_to_depot operational semantics")
        if op.fleet_model != "homogeneous_multi_compartment":
            raise ValueError("C0 requires a homogeneous multi-compartment fleet")
        if op.capacity_model != "shared_total":
            raise ValueError("C0 supports shared_total capacity only")
        if not op.single_trip or op.allow_reload:
            raise ValueError("C0 requires single_trip=True and allow_reload=False")
        if op.quality_clock != "service_finish_to_return_arrival":
            raise ValueError("C0 quality clock must start at service_finish")
        if op.shared_vehicle_capacity <= 0:
            raise ValueError("shared vehicle capacity must be positive")

        if (self.units.distance_km_per_unit <= 0
                or self.units.hours_per_time_unit <= 0
                or self.units.speed_kmph <= 0):
            raise ValueError("all unit scale values must be positive")

        th = self.thermal
        classes = th.supported_temp_classes
        if classes != tuple(range(len(classes))):
            raise ValueError("temperature classes must be contiguous and zero based")
        zone_fields = (
            th.target_temperature_c,
            th.hard_temperature_bounds_c,
            th.heat_transfer_per_hour,
            th.cooling_power_kw,
            th.cooling_rate_c_per_hour,
            th.door_heat_kwh,
            th.thermal_capacity_kwh_per_c,
            th.dispatch_preconditioning_energy_kwh,
        )
        if any(len(values) != len(classes) for values in zone_fields):
            raise ValueError("all thermal zone fields must match supported_temp_classes")
        for idx, bounds in enumerate(th.hard_temperature_bounds_c):
            low, high = bounds
            if not low < th.target_temperature_c[idx] < high:
                raise ValueError("zone target must lie strictly inside its hard bounds")
        if any(value < 0 for value in th.heat_transfer_per_hour):
            raise ValueError("heat-transfer coefficients must be non-negative")
        if any(value < 0 for value in th.cooling_power_kw + th.cooling_rate_c_per_hour):
            raise ValueError("cooling power/rate must be non-negative")
        if any(value < 0 for value in th.door_heat_kwh):
            raise ValueError("door heat must be non-negative")
        if any(value <= 0 for value in th.thermal_capacity_kwh_per_c):
            raise ValueError("thermal capacities must be positive")
        if any(value < 0 for value in th.dispatch_preconditioning_energy_kwh):
            raise ValueError("preconditioning energy must be non-negative")
        if len(th.cop_parameters) != 3 or min(th.cop_parameters[0], th.cop_parameters[2]) <= 0:
            raise ValueError("COP parameters must be (positive base, slope, positive minimum)")

        q = self.quality
        quality_fields = (
            q.reference_rate_per_hour,
            q.activation_over_r_kelvin,
            q.unsalable_quality_threshold,
            q.product_value,
        )
        if any(len(values) != len(classes) for values in quality_fields):
            raise ValueError("all quality fields must match supported_temp_classes")
        if any(value < 0 for value in q.reference_rate_per_hour):
            raise ValueError("quality rates must be non-negative")
        if any(value < 0 for value in q.activation_over_r_kelvin):
            raise ValueError("Arrhenius activation-over-R values must be non-negative")
        if q.reference_temperature_k <= 0:
            raise ValueError("reference temperature must be positive Kelvin")
        if any(not 0 < value <= 1 for value in q.unsalable_quality_threshold):
            raise ValueError("unsalable thresholds must lie in (0, 1]")
        if any(value <= 0 for value in q.product_value):
            raise ValueError("product values must be positive")

        obj = self.objective
        if min(obj.distance_scale, obj.quality_scale, obj.energy_scale) <= 0:
            raise ValueError("objective scales must be positive")
        if min(obj.lambda_quality, obj.lambda_energy) < 0:
            raise ValueError("objective weights must be non-negative")

        if not self.parameter_provenance:
            raise ValueError("parameter provenance must not be empty")
        seen = set()
        for item in self.parameter_provenance:
            if item.parameter_path in seen:
                raise ValueError("duplicate provenance path: %s" % item.parameter_path)
            seen.add(item.parameter_path)
            if item.source_type not in _ALLOWED_SOURCE_TYPES:
                raise ValueError("invalid source_type for %s" % item.parameter_path)
            if not item.source_reference.strip():
                raise ValueError("missing source reference for %s" % item.parameter_path)
            if item.sensitivity_low > item.sensitivity_high:
                raise ValueError("invalid sensitivity interval for %s" % item.parameter_path)
            if not item.unit.strip():
                raise ValueError("missing unit for %s" % item.parameter_path)
            section_name, field_name = item.parameter_path.split(".", 1)
            section = getattr(self, section_name, None)
            if section is None or not hasattr(section, field_name):
                raise ValueError("unknown provenance parameter path: %s"
                                 % item.parameter_path)
            values = _flatten_numeric(getattr(section, field_name))
            if not values or any(
                    value < item.sensitivity_low or value > item.sensitivity_high
                    for value in values):
                raise ValueError("default lies outside sensitivity interval for %s"
                                 % item.parameter_path)

    @property
    def contract_hash(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_manifest(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        # asdict keeps tuple fields as tuples; validate_contract_manifest expects
        # JSON-style lists, so normalize provenance before the in-memory round-trip.
        result["parameter_provenance"] = list(result["parameter_provenance"])
        result["contract_hash"] = self.contract_hash
        return result

    @classmethod
    def from_manifest(cls, data):
        """从序列化 manifest（`to_manifest` 输出 / JSON 文件）重建 contract 并校验 hash。

        严格结构校验 + 重算 contract_hash 与文件自报一致，防止字段漂移或缺失。
        JSON 把 tuple 序列化成 list，这里把 thermal/quality 的 list 规整回 tuple，
        使 `validate()` 的 tuple 相等性与长度检查按原始语义执行。
        """
        validate_contract_manifest(data)

        def _as_tuple(v):
            return tuple(_as_tuple(x) for x in v) if isinstance(v, list) else v

        contract = cls(
            schema_version=data['schema_version'],
            operational=OperationalConfig(**data['operational']),
            units=UnitScale(**data['units']),
            thermal=ThermalConfig(**{k: _as_tuple(v) for k, v in data['thermal'].items()}),
            quality=QualityConfig(**{k: _as_tuple(v) for k, v in data['quality'].items()}),
            objective=ColdChainObjectiveConfig(**data['objective']),
            parameter_provenance=tuple(
                ParameterProvenance(**p) for p in data['parameter_provenance']),
        )
        contract.validate()
        declared = data.get('contract_hash')
        if declared and contract.contract_hash != declared:
            raise ValueError("contract hash mismatch: recomputed=%s file=%s"
                             % (contract.contract_hash[:12], declared[:12]))
        return contract


def default_pilot_contract() -> ColdChainContract:
    """Return the explicit pilot contract used before literature calibration."""

    contract = ColdChainContract()
    contract.validate()
    return contract


def _require_fields(section, data, dataclass_type):
    if not isinstance(data, dict):
        raise ValueError("%s must be a dict" % section)
    fields = set(dataclass_type.__dataclass_fields__)
    missing = fields - set(data)
    extra = set(data) - fields
    if missing:
        raise ValueError("%s missing fields: %s" % (section, sorted(missing)))
    if extra:
        raise ValueError("%s unknown fields: %s" % (section, sorted(extra)))


def validate_contract_manifest(data):
    """Strict structural validation of a serialized cold-chain contract manifest.

    拒绝未知/缺失字段、错误 schema、非 64-hex contract_hash。参数值本身由
    `ColdChainContract.validate()` 在重建后校验。
    """
    if not isinstance(data, dict):
        raise ValueError("contract manifest must be a dict")
    if data.get('schema_version') != SCHEMA_VERSION:
        raise ValueError("unsupported cold-chain schema_version: %s" % data.get('schema_version'))
    allowed_top = {'schema_version', 'operational', 'units', 'thermal', 'quality',
                   'objective', 'parameter_provenance', 'contract_hash'}
    extra_top = set(data) - allowed_top
    if extra_top:
        raise ValueError("contract manifest unknown top-level fields: %s" % sorted(extra_top))
    for key, dt in (('operational', OperationalConfig), ('units', UnitScale),
                    ('thermal', ThermalConfig), ('quality', QualityConfig),
                    ('objective', ColdChainObjectiveConfig)):
        if key not in data:
            raise ValueError("contract manifest missing section: %s" % key)
        _require_fields(key, data[key], dt)
    pp = data.get('parameter_provenance')
    if not isinstance(pp, list) or not pp:
        raise ValueError("parameter_provenance must be a non-empty list")
    for item in pp:
        _require_fields('parameter_provenance', item, ParameterProvenance)
    if 'contract_hash' not in data:
        raise ValueError("contract manifest missing contract_hash")
    ch = data['contract_hash']
    if not (isinstance(ch, str) and len(ch) == 64
            and all(c in '0123456789abcdef' for c in ch)):
        raise ValueError("contract_hash must be a 64-hex string")
    return None


def load_coldchain_contract(path):
    """从 JSON 文件加载并校验 contract（重算 contract_hash 与文件自报一致）。"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return ColdChainContract.from_manifest(data)


def write_pilot_contract(path):
    """把当前 pilot contract 显式序列化为 JSON（不依赖 Python 默认值）。"""
    data = default_pilot_contract().to_manifest()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


@dataclass(frozen=True)
class ObjectiveProfile:
    """具名 cold-chain 目标 profile（scale + λ），用于冻结 O0-CC 主诊断口径。

    scale_source ∈ {'pilot', 'devmean'}。devmean 归一化的 scale 必须来自独立 DEV-CAL
    （不能是 O0-CC 判定用的 VAL），dev_statistics 记录均值/std/median/IQR、实例 ID、
    数据 hash 与场景权重（决策 1 限定）。
    """

    name: str
    distance_scale: float
    quality_scale: float
    energy_scale: float
    lambda_quality: float
    lambda_energy: float
    scale_source: str = 'pilot'
    dev_statistics: dict = None

    def __post_init__(self):
        if self.scale_source not in ('pilot', 'devmean'):
            raise ValueError("scale_source must be 'pilot' or 'devmean'")
        if min(self.distance_scale, self.quality_scale, self.energy_scale) <= 0:
            raise ValueError("objective scales must be positive")
        if min(self.lambda_quality, self.lambda_energy) < 0:
            raise ValueError("objective weights must be non-negative")

    def objective_config(self) -> ColdChainObjectiveConfig:
        return ColdChainObjectiveConfig(
            distance_scale=self.distance_scale,
            quality_scale=self.quality_scale,
            energy_scale=self.energy_scale,
            lambda_quality=self.lambda_quality,
            lambda_energy=self.lambda_energy,
        )

    @property
    def profile_hash(self) -> str:
        payload = json.dumps({
            'name': self.name,
            'distance_scale': self.distance_scale,
            'quality_scale': self.quality_scale,
            'energy_scale': self.energy_scale,
            'lambda_quality': self.lambda_quality,
            'lambda_energy': self.lambda_energy,
            'scale_source': self.scale_source,
            'dev_statistics': self.dev_statistics,
        }, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def to_manifest(self) -> dict:
        return {
            'name': self.name,
            'distance_scale': self.distance_scale,
            'quality_scale': self.quality_scale,
            'energy_scale': self.energy_scale,
            'lambda_quality': self.lambda_quality,
            'lambda_energy': self.lambda_energy,
            'scale_source': self.scale_source,
            'dev_statistics': self.dev_statistics,
            'profile_hash': self.profile_hash,
        }


def apply_objective_profile(contract: ColdChainContract,
                            profile: ObjectiveProfile) -> ColdChainContract:
    """用具名 profile 的 objective 覆盖 contract 的 objective，返回新 contract。"""
    contract.validate()
    return replace(contract, objective=profile.objective_config())


def default_pilot_profile() -> ObjectiveProfile:
    return ObjectiveProfile('pilot', 1.0, 1.0, 1.0, 1.0, 0.1, 'pilot')


def make_devmean_profile(name, statistics, lambda_quality=1.0, lambda_energy=1.0):
    """从 DEV-CAL 统计（含各分量算术均值）构造 devmean 归一化 profile。"""
    required = ('distance_mean', 'quality_mean', 'energy_mean')
    for key in required:
        if key not in statistics:
            raise ValueError("dev statistics missing: %s" % key)
    return ObjectiveProfile(
        name=name,
        distance_scale=float(statistics['distance_mean']),
        quality_scale=float(statistics['quality_mean']),
        energy_scale=float(statistics['energy_mean']),
        lambda_quality=float(lambda_quality),
        lambda_energy=float(lambda_energy),
        scale_source='devmean',
        dev_statistics=dict(statistics),
    )


def _flatten_numeric(value) -> tuple[float, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(number for item in value for number in _flatten_numeric(item))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (float(value),)
    return ()


def time_to_hours(duration_time_units: float, units: UnitScale) -> float:
    if duration_time_units < 0:
        raise ValueError("duration cannot be negative")
    return float(duration_time_units) * units.hours_per_time_unit


def distance_to_km(distance_units: float, units: UnitScale) -> float:
    if distance_units < 0:
        raise ValueError("distance cannot be negative")
    return float(distance_units) * units.distance_km_per_unit


def power_duration_to_kwh(power_kw: float, duration_h: float) -> float:
    if power_kw < 0 or duration_h < 0:
        raise ValueError("power and duration must be non-negative")
    return float(power_kw) * float(duration_h)


def joules_to_kwh(energy_joules: float) -> float:
    if energy_joules < 0:
        raise ValueError("energy cannot be negative")
    return float(energy_joules) / 3_600_000.0


def compute_coldchain_cost(
    distance_cost: float,
    quality_loss: float,
    energy_kwh: float,
    objective: ColdChainObjectiveConfig,
) -> float:
    """Compute the normalized scalar objective from auditable raw components."""

    if min(distance_cost, quality_loss, energy_kwh) < 0:
        raise ValueError("cold-chain objective components must be non-negative")
    if min(objective.distance_scale, objective.quality_scale, objective.energy_scale) <= 0:
        raise ValueError("cold-chain objective scales must be positive")
    return (
        distance_cost / objective.distance_scale
        + objective.lambda_quality * quality_loss / objective.quality_scale
        + objective.lambda_energy * energy_kwh / objective.energy_scale
    )
