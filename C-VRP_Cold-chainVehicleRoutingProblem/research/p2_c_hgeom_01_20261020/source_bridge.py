"""Execute frozen routing definitions without importing model/training modules.

AST nodes are compiled unchanged from their frozen source, not translated. Only
explicitly listed classes/functions and required type aliases are executable.
No source-day loader or model scorer is exported by this bridge.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import sys
import types
import typing

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(*, expected_hashes=None):
    import numpy as np
    reg = json.loads((PACKAGE / 'registration.json').read_text(encoding='utf-8'))
    frozen = (reg['seven_elements']['identity_and_delivery']['existing_source_hashes']
              if expected_hashes is None else expected_hashes)
    paths = {
        'ac': ROOT / 'scripts/simulation/action_contract.py',
        'ref': ROOT / 'scripts/experimental/p2_pos01_reference.py',
        'shared': ROOT / 'research/p2_c_camr_static_01_20261019/camr_shared.py',
    }
    selections = {
        'ac': None,  # all pure function/class nodes, no import/entry statements
        'ref': {'Order', 'Vehicle', 'PublicView', 'finite', 'identity', 'capacity_ok',
                'public_env', 'route_check', 'plans_from_view', 'mutable_ids', 'remove_mask'},
        'shared': {'shared_allowed', 'legal_actions_shared', 'plans_of', 'remove_mask',
                   'apply', 'signature', 'anchor_of', 'check_plan', 'suffix_first', 'unit_key'},
    }
    modules, receipt = {}, {}
    for name, path in paths.items():
        rel = path.relative_to(ROOT.parent).as_posix()
        if digest(path) != frozen[rel]:
            raise ValueError('IDENTITY_MISMATCH: ' + rel)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and (selections[name] is None or n.name in selections[name])]
        module = types.ModuleType('_hgeom_frozen_' + name)
        sys.modules[module.__name__] = module
        ns = module.__dict__
        ns.update(vars(typing))
        ns.update(np=np, math=math, hashlib=hashlib, json=json,
                  dataclass=dataclasses.dataclass, field=dataclasses.field,
                  replace=dataclasses.replace, asdict=dataclasses.asdict,
                  fields=dataclasses.fields, is_dataclass=dataclasses.is_dataclass,
                  SimpleNamespace=types.SimpleNamespace, CAPACITY_MARGIN=1e-4, EVENT_TOL=1e-6)
        if name == 'ref':
            ns.update(build_vehicle_plans=modules['ac'].build_vehicle_plans)
        if name == 'shared':
            ns.update(ac=modules['ac'], ref=modules['ref'])
        prefix = ast.parse('from __future__ import annotations').body
        if name == 'ac':
            # Required alias is the original assignment; VehiclePlan precedes it.
            body = []
            for n in nodes:
                body.append(n)
                if n.name == 'VehiclePlan':
                    body += [a for a in tree.body if isinstance(a, ast.Assign)
                             and any(isinstance(t, ast.Name) and t.id == 'FleetPlan'
                                     for t in a.targets)]
        else:
            body = nodes
        exec(compile(ast.Module(body=prefix + body, type_ignores=[]), str(path), 'exec'), ns)
        modules[name] = module
        receipt[rel] = {'source_sha256': digest(path), 'definitions': [n.name for n in nodes],
                        'ast_sha256': hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[]),
                                                              include_attributes=False).encode()).hexdigest()}
    return modules['ac'], modules['ref'], modules['shared'], receipt
