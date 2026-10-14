"""CPU engineering regressions; no online effect inference."""
import copy
import json
import math
from pathlib import Path
import pickle
import sys
import time
import unittest
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import public_fleet as pf
SCRIPTS=Path(sys.argv[sys.argv.index('--scripts')+1]) if '--scripts' in sys.argv else Path('D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts')
if '--scripts' in sys.argv:
    j=sys.argv.index('--scripts');del sys.argv[j:j+2]
pf.bootstrap(SCRIPTS)

class PickupReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet=pickle.loads((HERE/'public_certificate_qa_v2/public_packet.pkl').read_bytes())
        cls.qa=json.loads((HERE/'public_certificate_qa_v2/result.json').read_text())
        cls.census=json.loads((HERE/'full_census_diagnostic_v1/result.json').read_text())
    def test_full_census_keys_and_coverage(self):
        rows=json.loads((HERE/'full_census_diagnostic_v1/per_event_vehicle.json').read_text())
        groups={};keys=set()
        for r in rows:
            k=(r['day'],r['event_id'],r['vehicle_id'])
            self.assertNotIn(k,keys);keys.add(k)
            groups.setdefault(k[:2],set()).add(r['vehicle_id'])
        self.assertEqual(len(groups),1132);self.assertEqual(len(rows),16980)
        self.assertTrue(all(v==set(range(15)) for v in groups.values()))
        self.assertTrue(all(any(r['day']==d and r['clock']==0 for r in rows) for d in range(4)))
        self.assertEqual(max(r['mutable_n'] for r in rows),9)
        self.assertEqual(max(r['owned_n'] for r in rows),10)
    def test_census_replay_and_registration_not_gain(self):
        r=self.census
        self.assertEqual(r['source_start'],r['source_end'])
        self.assertTrue(r['independent_terminal_replay']['all_evaluations_equal'])
        self.assertFalse(r['contract_env_speed_consistent'])
        self.assertFalse(r['stable_online_gain_established'])
        self.assertFalse(r['pickup_impossibility_proven'])
        self.assertEqual(r['protocol']['data_seed'],2026120201)
        self.assertEqual(r['protocol']['protected_final_seeds'],[20260927,20260928])
    def test_archived_hidden_future_mutation(self):
        self.assertTrue(self.qa['future_mutation_packet_invariant'])
        self.assertTrue(self.qa['future_mutation_choice_invariant'])
        self.assertFalse(self.qa['full_future_day_used_for_selection'])
        self.assertFalse(self.qa['formal_online_gain_established'])
    def test_unit_mismatch_and_private_future_rejected(self):
        from coldchain_contract import ColdChainContractV2
        p=copy.deepcopy(self.packet);p['contract']=ColdChainContractV2();p['contract_hash']=p['contract'].contract_hash
        with self.assertRaisesRegex(ValueError,'speed'):pf.validate_public(p)
        p=copy.deepcopy(self.packet);p['dataset']['private_future']=[]
        with self.assertRaises(ValueError):pf.validate_public(p)
    def test_committed_destination_and_duplicate_service(self):
        plan=copy.deepcopy(self.packet['plan']);plan['0']=[20,43,17]
        with self.assertRaisesRegex(ValueError,'committed'):pf.certificate(self.packet,plan)
        plan=copy.deepcopy(self.packet['plan']);plan['1']=[20]
        with self.assertRaisesRegex(ValueError,'commitment'):pf.certificate(self.packet,plan)
        plan=copy.deepcopy(self.packet['plan']);plan['0'][0]=True
        with self.assertRaisesRegex(ValueError,'integer'):pf.certificate(self.packet,plan)
    def test_actual_multiveh_certificates_keep_own_incumbent(self):
        r=pf.solve_public(copy.deepcopy(self.packet),SCRIPTS)
        self.assertEqual(len(r['candidates']),6)
        self.assertEqual(r['selected_plan'],self.packet['plan'])
        self.assertTrue(all(c['certificate']['all_fleet_energy_accounted'] for c in r['candidates']))
        self.assertTrue(any(c['plan']['1'] for c in r['candidates']))
        self.assertEqual(r['source_start'],r['source_end'])
        self.assertAlmostEqual(r['selected_known']['utility'],75.48180642724037)
    def test_timeout_retains_incumbent_and_never_installs(self):
        r=pf.bounded_select(copy.deepcopy(self.packet),SCRIPTS,seconds=.001)
        self.assertTrue(r['fallback']);self.assertTrue(r['timeout'])
        self.assertEqual(r['selected_plan'],self.packet['plan'])
        rp=SimpleNamespace(_plan={99:[123]})
        self.assertFalse(pf.install_in_live(rp,self.packet,r));self.assertEqual(rp._plan,{99:[123]})
    def test_strict_clock_bool_and_qualification(self):
        for x in [True,float('nan'),float('inf'),'1']:
            with self.assertRaises(ValueError):pf.bounded_select(self.packet,SCRIPTS,seconds=x)
            with self.assertRaises(ValueError):pf.bounded_select(self.packet,SCRIPTS,started=x)
        for q in ['false',False,1]:
            r=dict(qualified=q,packet_sha=pf.digest(self.packet),deadline_monotonic=time.perf_counter()+1)
            self.assertFalse(pf.install_in_live(SimpleNamespace(),self.packet,r))
        r=dict(qualified=True,packet_sha=pf.digest(self.packet),deadline_monotonic=True)
        self.assertFalse(pf.install_in_live(SimpleNamespace(),self.packet,r))

if __name__=='__main__':unittest.main(verbosity=2)
