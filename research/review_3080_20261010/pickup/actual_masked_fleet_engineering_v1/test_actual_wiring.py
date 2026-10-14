"""Actual archived model evidence and causal/physical/budget regressions."""
import copy
import io
import json
import math
from pathlib import Path
import pickle
import queue
import sys
import time
from types import SimpleNamespace
import unittest
HERE=Path(__file__).resolve().parent
SCRIPTS=Path(sys.argv[sys.argv.index('--scripts')+1]) if '--scripts' in sys.argv else Path('D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts')
if '--scripts' in sys.argv:
    j=sys.argv.index('--scripts');del sys.argv[j:j+2]
sys.path.insert(0,str(HERE));import masked_fleet as mf
import fair_wiring as fw
import resident_actor as ra
import public_fleet as pf
pf.bootstrap(SCRIPTS)
from audit_io import decode
from scenario_saa import make_c0_contract_v2

class WiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet=pickle.loads((HERE.parent/'public_certificate_qa_v2/public_packet.pkl').read_bytes())
        cls.actual=decode(json.loads((HERE/'actual_qa_v1/result.json').read_text(encoding='utf-8')))
        import numpy as np
        cls.hist=[(np.array([.5,1.,2.]),np.array([0,1,2]))]
    def test_saved_actual_encoder_decoder_and_random234(self):
        r=self.actual;self.assertTrue(r['full_encoder_decoder_used']);self.assertTrue(r['random_params_differ'])
        rows={x['mode']:x for x in r['rows']}
        for mode in rows:
            p=rows[mode]['proposal'];self.assertEqual(p['provider_identity']['param_count'],234)
            self.assertEqual(p['features_shape'],[1,4,3]);self.assertTrue(p['retained_edges_preserved'])
            self.assertEqual(rows[mode]['selected'],self.packet['plan']);self.assertFalse(p['formal_online_gain_established'])
        self.assertNotEqual(rows['random']['proposal']['logits_sha256'],rows['pretrained']['proposal']['logits_sha256'])
    def test_topology_committed_public_feature_dimension(self):
        v,pool,ids,loc,inst,routes,comm=mf.topology(self.packet)
        self.assertEqual(set(pool),{20,17});self.assertEqual(comm,{43});self.assertEqual(inst['coords'].shape,(4,2))
        self.assertEqual(set(ids),{43,20,17})
        mutated=copy.deepcopy(self.packet);mutated['dataset']['private_future']=[]
        with self.assertRaises(ValueError):mf.topology(mutated)
    def test_strict_mask_guide_deadline(self):
        for name in ('mask_fraction','guide_weight','deadline'):
            for x in (True,'1',math.nan,math.inf):
                with self.assertRaises(ValueError):mf.propose(None,self.packet,SCRIPTS,**{name:x})
    def test_same_explicit_history_and_unit_contract(self):
        c=make_c0_contract_v2()
        rs=[fw.build_matched_replanner(n,SCRIPTS,self.hist,c,710.65,0.) for n in ('own_uncond_dynamic','strong_rolling')]
        self.assertEqual(rs[0].fair_identity['history_sha256'],rs[1].fair_identity['history_sha256'])
        self.assertIs(rs[0].hist,self.hist);self.assertIs(rs[1].hist,self.hist)
        class Sampler:
            def sample(self,*unused):return [[]]*10
        s=fw.build_matched_replanner('saa',SCRIPTS,self.hist,c,710.65,0.,sampler=Sampler())
        self.assertIs(s.hist,self.hist);self.assertFalse(s.anytime_vote);self.assertEqual(s.K,10)
        for k in (True,10.,9):
            with self.assertRaises(ValueError):fw.build_matched_replanner('strong_rolling',SCRIPTS,self.hist,c,700.,0.,k=k)
        with self.assertRaises(ValueError):fw.build_matched_replanner('strong_rolling',SCRIPTS,None,c,700.,0.)
        for x in (True,'10',math.nan):
            with self.assertRaises(ValueError):fw.build_matched_replanner('strong_rolling',SCRIPTS,self.hist,c,700.,0.,uniform_limit=x)
    def test_late_synchronous_admission_rolls_back_own_plan(self):
        class Fake:
            def __init__(self):self._accepted=set();self._rejected=set();self._plan={0:[1]}
            def _reset_if_new(self,i):pass
            def export_state(self):return dict(accepted=set(self._accepted),rejected=set(self._rejected),plan=copy.deepcopy(self._plan))
            def restore_state(self,s):self._accepted=s['accepted'];self._rejected=s['rejected'];self._plan=s['plan']
            def on_reveal(self,*args):self._accepted.add(2);self._plan[0].append(2);time.sleep(.004)
        import numpy as np
        env=SimpleNamespace(get_reserved_customers=lambda _:set(),reveal_time=np.array([[0,0,1.]]),event_id=4)
        r=fw.uniform_per_admission(Fake,SCRIPTS)(uniform_limit=.001)
        r.on_reveal(env,0,1.,[],np.zeros(3,dtype=bool),[2])
        self.assertEqual(r._plan,{0:[1]});self.assertEqual(r._accepted,set());self.assertEqual(r._rejected,{2})
        self.assertTrue(r.admission_audit[0]['fallback'])
    def test_resident_timeout_kills_and_preserves_incumbent(self):
        class Q:
            def get(self,timeout):raise queue.Empty
        a=ra.ResidentActor.__new__(ra.ResidentActor);a.stopped=False;a.lines=Q();a.process=SimpleNamespace(stdin=io.StringIO())
        killed=[]
        a.close=lambda:(killed.append(True),setattr(a,'stopped',True))
        r=a.select(self.packet,time.perf_counter(),.001)
        self.assertTrue(killed);self.assertTrue(r['timeout']);self.assertEqual(r['selected_plan'],self.packet['plan'])
        for x in (True,'1',math.nan,math.inf):
            with self.assertRaises(ValueError):a.select(self.packet,x)
    def test_late_install_never_commits(self):
        r=dict(qualified=True,selected_plan=self.packet['plan'],packet_sha=pf.digest(self.packet),vehicle_ids=[0,1],deadline_monotonic=time.perf_counter()-1)
        rp=SimpleNamespace(_plan={9:[99]});self.assertFalse(pf.install_in_live(rp,self.packet,r));self.assertEqual(rp._plan,{9:[99]})
    def test_actual_uniform_full_day_replay_and_histories(self):
        r=decode(json.loads((HERE/'wiring_qa_v1/result.json').read_text(encoding='utf-8')))
        self.assertTrue(r['matching_history_verified']);self.assertFalse(r['full_online_fresh_gain_established'])
        for row in r['rows']:
            self.assertTrue(row['evaluation']['hard_feasible']);self.assertEqual(len(row['admissions']),80)
            self.assertTrue(row['independent_replay']['all_evaluations_equal'])
            self.assertTrue(all(x['actual_certified'] for x in row['admissions']))
            self.assertTrue(all(x['wall_seconds']<=10 for x in row['admissions']))
            self.assertFalse(row['identity']['legacy_compute_kill_qualified'])
    def test_actual_resident_new_timeout_evidence(self):
        import strict_online_env,coldchain_contract,coldchain_state
        from dataclasses import is_dataclass
        classes={c.__module__+'.'+c.__name__:c for m in (strict_online_env,coldchain_contract,coldchain_state)
                 for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
        r=decode(json.loads((HERE/'resident_qa_v1/result.json').read_text(encoding='utf-8')),classes)
        self.assertTrue(r['selection']['qualified']);self.assertTrue(r['timeout_worker_reaped']);self.assertTrue(r['timeout_keeps_own'])
        self.assertEqual(r['actual_ready']['provider_identity']['actual_param_leaves'],234)
        self.assertEqual(r['selection']['source_end'],r['actual_ready']['source_start'])
        self.assertFalse(r['new_day_or_online_gain'])
    def test_resident_actual_end_weight_identity_required(self):
        a=ra.ResidentActor.__new__(ra.ResidentActor);a.stopped=False;a.process=SimpleNamespace(stdin=io.StringIO())
        a.ready=dict(source_start={'s':'h'},checkpoint_sha256='w',provider_identity={'p':234})
        r=dict(qualified=True,selected_plan=self.packet['plan'],source_end={'s':'h'},checkpoint_sha256='w',
               provider_identity={'p':234},packet_sha=pf.digest(self.packet))
        a.lines=queue.Queue();a.lines.put(dict(kind='response',result=ra.wire(r)))
        self.assertFalse(a.select(self.packet,time.perf_counter())['qualified'])
        r['checkpoint_end_sha256']='w';a.lines.put(dict(kind='response',result=ra.wire(r)))
        self.assertTrue(a.select(self.packet,time.perf_counter())['qualified'])

if __name__=='__main__':unittest.main()
