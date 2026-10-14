"""Explicit fixed-schedule I3 training engine, not a registered experiment.

No evaluation data, checkpoint selection, teacher collection or defaults.
Caller validates private archive/source/history context, initializes a sealed
model and supplies an exact ordered state list/settings. All states remain in
the schedule; invalid/terminal teachers receive reconstruction-only updates.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ('models', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from i3_actor_draws import draw_single_step_sets, validated_actor_arrays
from i3_counterfactual_teacher import record_identity, scenario_margin, teacher_preference
from i3_decision_objective import decision_loss, validate_decision_inputs
from i3_optimizer import (create_optimizer, make_gradient_step, apply_gradient_step,
    make_reconstruction_step, apply_reconstruction_step, parameter_hash)
from maskco_scenario import (MASK_TOKEN, PAD_TOKEN, M_MAX, VOCAB_SIZE,
                            order_to_token, clock_bin_of, clock_support_mask)


@dataclass(frozen=True)
class TrainingSettings:
    steps: int
    K: int
    independent_sets: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    reconstruction_weight: float
    tau: float
    weight_cap: float
    baseline_decay: float
    schedule_seed: int
    actor_seed: int
    label_seed: int
    objective: str

    def validate(self):
        if (any(type(v) is not int or v <= 0 for v in
                (self.steps, self.K, self.independent_sets)) or
                any(type(v) is not int or v < 0 for v in
                    (self.schedule_seed, self.actor_seed, self.label_seed)) or
                self.objective not in ('decision_reconstruction', 'reconstruction_control')):
            raise ValueError('Explicit positive sizes/seeds/objective required')
        positive = (self.learning_rate, self.max_grad_norm, self.reconstruction_weight,
                    self.tau, self.weight_cap)
        if (not all(np.isfinite(v) and v > 0 for v in positive) or
                not np.isfinite(self.weight_decay) or self.weight_decay < 0 or
                not np.isfinite(self.baseline_decay) or not 0 <= self.baseline_decay < 1):
            raise ValueError('Invalid fixed optimizer/loss settings')


def reconstruction_example(sampler, visible, true_day, rng):
    """Private future only supplies labels; actor input is sealed visible tokens.

    Full masked 200 slots at a fixed 400 sequence length, including zero-future
    states. Shuffle target slots with a separate RNG, not the actor draw stream.
    """
    prefix = sampler._visible_tokens(visible)
    future = [order_to_token(o) for o in true_day if o.reveal > visible.clock + 1e-6]
    if len(future) > M_MAX:
        raise ValueError('True future count exceeds registered capacity')
    rng.shuffle(future)
    x = np.full((1, 2 * M_MAX), PAD_TOKEN, dtype=np.int32)
    x[0, :len(prefix)] = prefix
    positions = np.arange(len(prefix), len(prefix) + M_MAX, dtype=np.int32)
    x[0, positions] = MASK_TOKEN
    targets = np.zeros_like(x)
    targets[0, positions[:len(future)]] = future
    return {'recon_x': x, 'recon_y': targets,
            'recon_n': np.array([len(future)], dtype=np.int32),
            'recon_cb': np.array([clock_bin_of(visible.clock)], dtype=np.int32)}


def train_fixed_schedule(sampler, entries, dataset, true_history, factory_for_day,
                         contract, booking_horizon, tw_speed, settings, *,
                         minimum_teacher_gap, use_jit=True, on_step=None):
    """Current-theta draws on FIXED teacher states; not a global policy gradient.

    Exactly one epoch permutation at a time; no sign/eligibility filtering.
    The lagged global EMA updates AFTER an eligible actor update only. Controls
    share schedule/init/optimizer/step count; they do not claim equal wall cost.
    """
    settings.validate()
    if not entries or sampler.iterative or sampler.m_max != M_MAX:
        raise ValueError('Nonempty fixed single-step training entries required')
    keys = [(e['day_id'], e['ordinal']) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate training states')
    for entry in entries:
        if entry['record'].record_hash != record_identity(entry['record']):
            raise ValueError('Private teacher record changed')
        if teacher_preference(*entry['branches'], minimum_gap=minimum_teacher_gap) != entry['preference']:
            raise ValueError('Teacher labels changed after archive verification')
        if entry['record'].visible.clock != entry['record'].snapshot['clock']:
            raise ValueError('Teacher generation/physical clocks disagree')
        if entry['day_id'] not in true_history:
            raise ValueError('Missing private reconstruction day')
    model = sampler.model
    initial = parameter_hash(model, frozen=False)
    frozen = parameter_hash(model, frozen=True)
    tx, opt_state, trainable = create_optimizer(model, learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay, max_grad_norm=settings.max_grad_norm)
    actor_step = make_gradient_step(tx, reconstruction_weight=settings.reconstruction_weight,
                                   use_jit=use_jit)
    recon_step = make_reconstruction_step(tx, reconstruction_weight=settings.reconstruction_weight,
                                          use_jit=use_jit)
    schedule_rng = np.random.default_rng(settings.schedule_seed)
    order, cursor, prior, logs = [], 0, 0., []
    for step in range(settings.steps):
        if cursor == len(order):
            order, cursor = schedule_rng.permutation(len(entries)).tolist(), 0
        entry = entries[order[cursor]]
        cursor += 1
        record, teacher = entry['record'], entry['preference']
        before = parameter_hash(model, frozen=False)
        label_rng = np.random.default_rng([settings.label_seed, step])
        recon = reconstruction_example(sampler, record.visible,
                                        true_history[entry['day_id']], label_rng)
        supported = bool(np.any(clock_support_mask(record.visible.clock) &
                                (np.arange(VOCAB_SIZE) != 0)))
        actor_enabled = (settings.objective == 'decision_reconstruction' and
                         teacher['decision_eligible'] and supported)
        old_prior, margins, packet = prior, None, None
        if actor_enabled:
            packet = draw_single_step_sets(sampler, record.visible,
                np.random.default_rng([settings.actor_seed, step]),
                independent_sets=settings.independent_sets, K=settings.K)
            margins = np.array([[scenario_margin(record, dataset,
                factory_for_day(entry['day_id']), contract, booking_horizon, tw_speed,
                scenes)['margin'] for scenes in packet['scenarios']]], dtype=np.float32)
            delta = np.array([teacher['loss_teacher_margin']], dtype=np.float32)
            eligible = np.array([True], dtype=bool)
            baseline = np.array([old_prior], dtype=np.float32)
            validate_decision_inputs(delta, margins, eligible, baseline)
            losses = np.asarray(decision_loss(delta, margins, eligible,
                settings.tau, settings.weight_cap), dtype=np.float32)
            arrays = validated_actor_arrays(sampler, packet)
            if not np.array_equal(arrays['actor_x'], recon['recon_x']):
                raise ValueError('Reconstruction input differs from generation condition')
            batch = dict(arrays, **recon, losses=losses, prior_baseline=baseline,
                         decision_eligible=eligible)
            stats, opt_state, trainable = apply_gradient_step(model, actor_step,
                opt_state, trainable, batch, sampled_trainable_hash=packet['sampled_trainable_hash'],
                sampled_frozen_hash=packet['sampled_frozen_hash'])
            prior = settings.baseline_decay * old_prior + (1 - settings.baseline_decay) * float(losses.mean())
        else:
            stats, opt_state, trainable = apply_reconstruction_step(model, recon_step,
                opt_state, trainable, recon, frozen_hash=frozen)
        row = {'step': step + 1, 'day_id': entry['day_id'], 'ordinal': entry['ordinal'],
            'record_hash': record.record_hash, 'teacher_valid': teacher['teacher_valid'],
            'teacher_delta': teacher['teacher_delta'],
            'decision_eligible': teacher['decision_eligible'], 'actor_update': bool(actor_enabled),
            'future_support': supported, 'prior_baseline': old_prior, 'next_baseline': prior,
            'estimated_margins': margins.tolist() if margins is not None else None,
            'draw_hash': packet['draw_hash'] if packet is not None else None,
            'trainable_before': before, 'trainable_after': parameter_hash(model, frozen=False),
            **stats}
        logs.append(row)
        if on_step is not None:
            on_step(row)
    return {'settings': asdict(settings), 'steps_completed': len(logs),
            'initial_trainable_hash': initial, 'final_trainable_hash': parameter_hash(model, frozen=False),
            'frozen_hash': frozen, 'train_keys': keys, 'logs': logs,
            'state_distribution': 'fixed teacher policy, not learner-collected',
            'online_gain_established': False}
