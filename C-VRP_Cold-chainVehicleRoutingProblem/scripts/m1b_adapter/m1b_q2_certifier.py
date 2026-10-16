#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""M1-b/Q2: the ENGINEERING CERTIFICATION ORCHESTRATOR.

Runs the gates in a FIXED order and produces a structured receipt.  The two hard gates are
recorded in SEPARATE, NON-SUBSTITUTABLE fields:

    QUALITY_LB_OK   -- Q2-3 (Omega10) + Q2-5 (analytic v2 quality lower bound)
    THERMAL_HARD_OK -- Q2-6 (state-anchored thermal replay vs hard_temperature_bounds_c)

An engineering composite pass requires EVERY gate to be independently true.  Any failure OR
unknown makes the composite False.  Neither gate may stand in for the other: the v2 quality
update contains no temperature term, so a quality pass says nothing about thermal feasibility,
and vice versa.

ORDER (each stage is a precondition of the next; nothing is evaluated speculatively):
    1 contract identity            (G2)
    2 immutable snapshot + aliasing (G3/G4)
    3 Omega11 return feasibility    (Q2-4)
    4 Omega10 initialization        (Q2-3)   [BEFORE any quality implementation is called]
    5 thermal replay hard gate      (Q2-6)
    6 analytic quality certificate  (Q2-5)
    7 post-state consistency        (G3)
    8 composite

WHAT THIS IS NOT: this is an ENGINEERING acceptance.  It is NOT production installability, NOT
scientific evidence, NOT L2, and it does NOT establish `FULL_INSTALLABILITY_ESTABLISHED`.
"""
from __future__ import annotations

import m1b_contract_identity as CI
import m1b_immutable_snapshot as IS
import m1b_quality_certificate as QC
import m1b_return_feasibility as RF
import m1b_thermal_gate as TG

SCHEMA = 'm1b_q2_engineering_receipt_v1'

ENGINEERING_CERTIFIED = 'M1B_Q2_ENGINEERING_CERTIFIED'
ENGINEERING_REFUSED = 'M1B_Q2_ENGINEERING_REFUSED'
NOT_ESTABLISHED = 'M1B_Q2_NOT_ESTABLISHED'

COMPOSITE_DISCLAIMER = (
    'engineering acceptance only: NOT production installability, NOT scientific evidence, NOT '
    'L2, and it does NOT establish FULL_INSTALLABILITY_ESTABLISHED')


class M1bQ2Certifier(object):
    def __init__(self, *, contract, registered_contract_hash=None, T_end, t_min=0.0,
                 epsilon_init=QC.EPSILON_INIT, epsilon_return=RF.EPSILON_RETURN,
                 budget=None, budget_is_synthetic=True):
        self.contract = contract
        self.registered_contract_hash = registered_contract_hash
        self.T_end = T_end
        self.t_min = t_min
        self.epsilon_init = epsilon_init
        self.epsilon_return = epsilon_return
        self.budget = budget
        self.budget_is_synthetic = budget_is_synthetic
        self.receipts = []

    # ------------------------------------------------------------------ main entry
    def certify(self, *, vehicles, served_mask, candidate_plans, clock, inst_idx,
                travel_fn, distance_fn, temp_class_of, quantity_of, service_time_of,
                initial_quality_by_node=None, active_zone_mask=None, lot_provider=None,
                quality_impl=None):
        receipt = {
            'schema': SCHEMA, 'proof_version': CI.CERTIFICATE_PROOF_VERSION,
            'inst_idx': int(inst_idx), 'clock': float(clock),
            'epsilon_init': float(self.epsilon_init), 'epsilon_return': float(self.epsilon_return),
            'budget_is_synthetic': bool(self.budget_is_synthetic),
            'QUALITY_LB_OK': None, 'THERMAL_HARD_OK': None,
            'composite_pass': False, 'final': NOT_ESTABLISHED,
            'disclaimer': COMPOSITE_DISCLAIMER,
            'gates': {},
        }

        # ---- 1 contract identity -------------------------------------------------
        ident, cverd, cdet = CI.validate_contract_for_quality_certificate(
            self.contract, registered_contract_hash=self.registered_contract_hash)
        receipt['contract_identity'] = ident
        receipt['gates']['contract_identity'] = {'verdict': cverd, 'detail': cdet}
        if cverd != CI.CONTRACT_IDENTITY_OK:
            receipt['final'] = ENGINEERING_REFUSED
            self.receipts.append(receipt)
            return False, receipt
        per_class, pcverd, pcdet = CI.check_quality_classes_are_certifiable(
            self.contract, epsilon_init=self.epsilon_init)
        receipt['gates']['quality_class_scope'] = {'verdict': pcverd, 'detail': pcdet}
        if pcverd != CI.CONTRACT_IDENTITY_OK:
            receipt['final'] = ENGINEERING_REFUSED
            self.receipts.append(receipt)
            return False, receipt

        # ---- 2 immutable snapshot + aliasing ------------------------------------
        snap, sverd, sdet = IS.build_frozen_snapshot(
            vehicles, served_mask, candidate_plans,
            initial_quality_by_node=initial_quality_by_node)
        receipt['gates']['snapshot'] = {'verdict': sverd, 'detail': sdet}
        if sverd != IS.SNAPSHOT_OK:
            receipt['final'] = ENGINEERING_REFUSED
            self.receipts.append(receipt)
            return False, receipt
        ali_ok, ali_det = IS.detect_aliasing(snap, vehicles, candidate_plans)
        receipt['gates']['aliasing'] = {
            'verdict': 'NO_ALIAS' if ali_ok else IS.SNAPSHOT_ALIAS_DETECTED, 'detail': ali_det}
        if not ali_ok:
            receipt['final'] = ENGINEERING_REFUSED
            self.receipts.append(receipt)
            return False, receipt
        receipt['candidate_plan_identity'] = snap['plan_identity']
        receipt['state_identity_before'] = snap['state_identity']

        # ---- 3 Omega11 return feasibility ---------------------------------------
        tfn = (lambda a, b: float(travel_fn(int(a), int(b))))
        ok11, v11, d11 = RF.certify_return_feasibility(
            snap, T_end=self.T_end, clock=clock, travel_fn=tfn,
            epsilon_return=self.epsilon_return)
        receipt['gates']['omega11_return_feasibility'] = {'verdict': v11, 'detail': d11}
        if not ok11:
            receipt['final'] = ENGINEERING_REFUSED
            self.receipts.append(receipt)
            return False, receipt

        # ---- 4 Omega10 initialization (BEFORE any quality implementation) --------
        lots = list(lot_provider(snap) if lot_provider is not None else _default_lot_provider(snap))
        ok10, v10, d10 = QC.check_omega10_initialization(
            lots, per_class, epsilon_init=self.epsilon_init)
        receipt['gates']['omega10_initialization'] = {'verdict': v10, 'detail': d10}
        if not ok10:
            # The quality implementation is NEVER called on this path: no callable is even
            # accepted as an argument to this stage.
            receipt['gates']['quality_impl_call'] = {
                'verdict': 'NOT_CALLED_OMEGA10_REFUSED',
                'detail': {'why': 'the initialization gate precedes the quality stage'}}
            receipt['final'] = ENGINEERING_REFUSED
            self.receipts.append(receipt)
            return False, receipt

        # ---- 5 thermal replay hard gate -----------------------------------------
        thermal_rows, thermal_verdicts = {}, []
        for vid in sorted(snap['vehicles']):
            fv = snap['vehicles'][vid]
            env, tverd, tdet = TG.replay_thermal_envelope(
                fv, contract=self.contract, inst_idx=inst_idx, clock=clock,
                travel_fn=tfn, distance_fn=(lambda a, b: float(distance_fn(int(a), int(b)))),
                temp_class_of=temp_class_of, quantity_of=quantity_of,
                service_time_of=service_time_of, active_zone_mask=active_zone_mask)
            thermal_rows[vid] = {'verdict': tverd, 'detail': tdet}
            thermal_verdicts.append(tverd)
        thermal_ok = all(v == TG.THERMAL_HARD_OK for v in thermal_verdicts) and bool(thermal_rows)
        receipt['gates']['thermal_replay'] = {
            'verdict': TG.THERMAL_HARD_OK if thermal_ok else _first_bad(thermal_verdicts),
            'detail': {'per_vehicle': thermal_rows},
            'continuous_time_hard_constraint': TG.CONTINUOUS_NOT_VERIFIED}
        receipt['THERMAL_HARD_OK'] = bool(thermal_ok)

        # ---- 6 analytic quality certificate -------------------------------------
        units = getattr(self.contract, 'units', None)
        hpu = getattr(units, 'hours_per_time_unit', None)
        qres, qverd, qdet = QC.certify_quality_lb(
            per_class, T_end=self.T_end, t_min=self.t_min, hours_per_time_unit=hpu,
            epsilon_init=self.epsilon_init)
        receipt['gates']['quality_lb'] = {'verdict': qverd, 'detail': qdet}
        receipt['QUALITY_LB_OK'] = bool(qverd == QC.QUALITY_LB_OK)
        # An OPTIONAL independently supplied implementation is consulted ONLY for record; it can
        # never override the analytic verdict, and it is called after Omega10 has already passed.
        if quality_impl is not None:
            try:
                impl_res = quality_impl(snap, qres)
                receipt['gates']['quality_impl_call'] = {
                    'verdict': 'CALLED_AFTER_OMEGA10_PASS',
                    'detail': {'result_repr': repr(impl_res)[:200],
                               'note': 'advisory only; the analytic verdict governs'}}
            except Exception as exc:                                    # noqa: BLE001
                receipt['gates']['quality_impl_call'] = {
                    'verdict': 'IMPL_RAISED', 'detail': {'exception': type(exc).__name__}}
                receipt['QUALITY_LB_OK'] = False

        # ---- 7 post-state consistency -------------------------------------------
        unch, uverd, udet = IS.verify_state_unchanged(snap, vehicles, served_mask,
                                                      candidate_plans)
        receipt['gates']['state_unchanged'] = {'verdict': uverd, 'detail': udet}
        receipt['state_identity_after'] = (udet or {}).get('after', snap['state_identity'])

        # ---- 8 composite ---------------------------------------------------------
        q_ok = receipt['QUALITY_LB_OK'] is True
        t_ok = receipt['THERMAL_HARD_OK'] is True
        all_gates = (cverd == CI.CONTRACT_IDENTITY_OK and pcverd == CI.CONTRACT_IDENTITY_OK
                     and sverd == IS.SNAPSHOT_OK and ali_ok and ok11 and ok10 and unch)
        receipt['composite_pass'] = bool(all_gates and q_ok and t_ok)
        receipt['final'] = ENGINEERING_CERTIFIED if receipt['composite_pass'] else ENGINEERING_REFUSED
        receipt['non_substitutable'] = {
            'QUALITY_LB_OK': receipt['QUALITY_LB_OK'],
            'THERMAL_HARD_OK': receipt['THERMAL_HARD_OK'],
            'rule': 'each gate is independently required; neither may substitute for the other; '
                    'unknown counts as failure'}
        self.receipts.append(receipt)
        return receipt['composite_pass'], receipt


def _first_bad(verdicts):
    for v in verdicts:
        if v != TG.THERMAL_HARD_OK:
            return v
    return TG.THERMAL_REPLAY_INCOMPLETE


def _default_lot_provider(snap):
    """Default: the cargo-manifest lots of the vehicles involved in the candidate plans.

    Deliberately does NOT touch unrevealed orders.  A caller with a richer but still public view
    (revealed orders the candidate references, plus committed visible orders) should pass its own
    `lot_provider`.
    """
    seen, out = set(), []
    for vid in sorted(snap['vehicles']):
        fv = snap['vehicles'][vid]
        if fv.plan is None:
            continue
        for lot in (getattr(fv.coldchain_state, 'cargo_manifest', ()) or ()):
            if int(lot.order_id) in seen:
                continue
            seen.add(int(lot.order_id))
            out.append(lot)
    return out
