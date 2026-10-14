import copy
import json
from pathlib import Path
import tempfile
import unittest
import new_run_gate_adapter as g


class NewRunGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        # Copy the immutable PROGRAM, not an invented fixture for its byte pin.
        original = Path('D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/research_20261010_continuation_v2/PROGRAM.json')
        (self.root/'PROGRAM.json').write_bytes(original.read_bytes())
        amend=original.parent/'review_followup_v3/CONTINUATION_V3_REGISTRATION.json'
        (self.root/'SCOPE_AMENDMENT.json').write_bytes(amend.read_bytes())
        self.identity = {k: self.ref(k+'.json', {'synthetic_role': k}) for k in g.ROLES}
        alpha = .05/(1*2)/2/(2*3)
        self.reg = {'schema':'prospective_formal_run_v1','stage':'confirmation','line':'delivery',
                    'program':{'path':'PROGRAM.json','sha256':g.sha(self.root/'PROGRAM.json')},
                    'scope_amendment':{'path':'SCOPE_AMENDMENT.json','sha256':g.SCOPE_AMENDMENT_SHA},
                    'family_index':1,'arms':['model','random','strong'],
                    'cohorts':[{'seed':s,'day_ids':list(range(40))} for s in [101,102,103]],
                    'contrasts':[{'id':a,'treatment':'model','control':a,'minimum_effect':.01,
                        'metric':'relative_complete_cost_reduction',
                        'control_role':('same_architecture_random' if a=='random' else 'qualified_strong_nonlearned_rolling')}
                        for a in ['random','strong']],
                    'statistics':{'two_sided_alpha':alpha,'bootstrap_replicates':24000,
                        'resampling_unit':'complete_day','paired':True,'bootstrap_seed':777,
                        'percentiles':[alpha/2*100,(1-alpha/2)*100]}}
        self.bundle={'schema':'formal_run_raw_package_v1','identities':self.identity,'days':[]}
        for s in [101,102,103]:
            for d in range(40):
                for a in self.reg['arms']:
                    prefix=f'{s}_{d}_{a}'
                    data={'schema':'complete_online_day_raw_v1','seed':s,'day_id':d,'arm':a,
                          'complete':True,'hard':True,'utility_or_cost':1.,
                          'identity_sha256':{k:v['sha256'] for k,v in self.identity.items()},
                          'event_journal':self.ref(prefix+'_events.json',{'synthetic':prefix}),
                          'model_call_records':self.ref(prefix+'_calls.json',{'synthetic':prefix}),
                          'runtime_process_records':self.ref(prefix+'_runtime.json',{'synthetic':prefix}),
                          'admissions':[{'entry_monotonic':1.,'submit_monotonic':1.5,'cap_s':10.,
                                         'own_incumbent_fallback':False,'excluded_internal_setup_s':0}]}
                    self.bundle['days'].append({'seed':s,'day_id':d,'arm':a,'raw':self.ref(prefix+'_raw.json',data)})

    def tearDown(self): self.tmp.cleanup()

    def ref(self,name,data):
        p=self.root/name;p.write_text(json.dumps(data),encoding='utf-8')
        return {'path':name,'sha256':g.sha(p)}

    def run_audit(self):
        self.ref('registration.json',self.reg)
        self.bundle['registration_sha256']=g.sha(self.root/'registration.json')
        self.ref('bundle.json',self.bundle)
        self.ref('registry.json',{'schema':'external_new_run_registry_v1',
            'registration_sha256':self.bundle['registration_sha256'],'program_sha256':g.PROGRAM_SHA,
            'scope_amendment_sha256':g.SCOPE_AMENDMENT_SHA,
            'identities':self.identity})
        return g.audit(self.root,self.root/'registration.json',self.root/'registry.json',self.root/'bundle.json')

    def mutate_raw(self,key,value):
        item=self.bundle['days'][0]; p=self.root/item['raw']['path']; d=g.load(p); d[key]=value
        item['raw']=self.ref(item['raw']['path'],d)

    def test_complete_package_is_subset_only(self):
        d=self.run_audit();self.assertTrue(d['registered_raw_byte_coverage_pass'])
        for k in ['independent_actual_model_activation_established','independent_process_kill_budget_proven',
                  'independent_full_trajectory_replay_established','formal_online_qualification_established',
                  'stable_maskco_core_online_gain','pre_data_registration_timing_established']:
            self.assertIs(d[k],False)

    def test_four_day_cannot_pass(self):
        self.reg['cohorts'][0]['day_ids']=list(range(4))
        with self.assertRaisesRegex(ValueError,'40 complete'):self.run_audit()

    def test_protected_seeds(self):
        for s in [20260927,20260928]:
            self.reg['cohorts'][0]['seed']=s
            with self.assertRaisesRegex(ValueError,'old final'):self.run_audit()

    def test_family_index_true_is_not_one(self):
        self.reg['family_index']=True
        with self.assertRaisesRegex(ValueError,'family index'):self.run_audit()

    def test_unadjusted95_rejected(self):
        self.reg['statistics']['two_sided_alpha']=.05
        with self.assertRaisesRegex(ValueError,'alpha allocation'):self.run_audit()

    def test_2000_reps_insufficient_tail(self):
        self.reg['statistics']['bootstrap_replicates']=2000
        with self.assertRaisesRegex(ValueError,'tail Monte Carlo'):self.run_audit()

    def test_invalid_percentile(self):
        self.reg['statistics']['percentiles']=[2.5,97.5]
        with self.assertRaisesRegex(ValueError,'percentile'):self.run_audit()

    def test_missing_actual_day(self):
        self.bundle['days'].pop()
        with self.assertRaisesRegex(ValueError,'raw coverage'):self.run_audit()

    def test_hard_true_string(self):
        self.mutate_raw('hard','true')
        with self.assertRaisesRegex(ValueError,'literal-hard'):self.run_audit()

    def test_nan_cost(self):
        self.mutate_raw('utility_or_cost',float('nan'))
        with self.assertRaisesRegex(ValueError,'nonfinite'):self.run_audit()

    def test_training_identity_missing(self):
        del self.identity['training_schedule']
        with self.assertRaisesRegex(ValueError,'identity roles'):self.run_audit()

    def test_changed_actual_source(self):
        (self.root/'source_closure.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):self.run_audit()

    def test_stored_identity_forgery(self):
        self.mutate_raw('identity_sha256',{})
        with self.assertRaisesRegex(ValueError,'role binding'):self.run_audit()

    def test_traversal(self):
        self.bundle['days'][0]['raw']['path']='../outside.json'
        with self.assertRaisesRegex(ValueError,'traversal'):self.run_audit()

    def test_late_no_fallback(self):
        self.mutate_raw('admissions',[{'entry_monotonic':1,'submit_monotonic':12,'cap_s':10,
                                      'own_incumbent_fallback':False,'excluded_internal_setup_s':0}])
        with self.assertRaisesRegex(ValueError,'late submission'):self.run_audit()

    def test_internal_setup_exclusion(self):
        self.mutate_raw('admissions',[{'entry_monotonic':1,'submit_monotonic':2,'cap_s':10,
                                      'own_incumbent_fallback':False,'excluded_internal_setup_s':.1}])
        with self.assertRaisesRegex(ValueError,'setup exclusion'):self.run_audit()

    def test_raw_boolean_day_id(self):
        self.mutate_raw('day_id',False)
        with self.assertRaisesRegex(ValueError,'integer IDs'):self.run_audit()

    def test_duplicate_json_key(self):
        p=self.root/'duplicate.json';p.write_text('{"hard":false,"hard":true}')
        with self.assertRaisesRegex(ValueError,'duplicate JSON'):g.load(p)

    def test_boolean_registered_day(self):
        self.reg['cohorts'][0]['day_ids'][0]=False
        with self.assertRaisesRegex(ValueError,'40 complete'):self.run_audit()

    def test_delivery_weaker_threshold(self):
        self.reg['contrasts'][0]['minimum_effect']=0
        with self.assertRaisesRegex(ValueError,'minimum>=1%'):self.run_audit()

    def test_missing_random_role(self):
        self.reg['contrasts'][0]['control_role']='proxy'
        with self.assertRaisesRegex(ValueError,'control registrations'):self.run_audit()

    def test_pickup_main_gate_weaker(self):
        self.reg['line']='pickup'
        self.reg['arms'].append('cond_hist')
        for c,role in zip(self.reg['contrasts'],['qualified_strong_nonlearned_rolling','same_init_reconstruction_control']):
            c['control_role']=role;c['metric']='complete_day_utility_difference'
        with self.assertRaisesRegex(ValueError,'minimum20'):self.run_audit()


class PickupMainRoleRegression(unittest.TestCase):
    setUp=NewRunGateTests.setUp
    tearDown=NewRunGateTests.tearDown
    ref=NewRunGateTests.ref
    run_audit=NewRunGateTests.run_audit
    def pickup_registration(self):
        self.reg['line']='pickup'
        for c,role,minimum in zip(self.reg['contrasts'],
            ['qualified_strong_nonlearned_rolling','same_init_reconstruction_control'],[20.,0.]):
            c.update(control_role=role,metric='complete_day_utility_difference',minimum_effect=minimum)

    def test_historical_cond_main_is_rejected(self):
        self.pickup_registration();self.reg['arms'].append('cond_hist')
        self.reg['contrasts'][0]['control_role']='cond_hist'
        with self.assertRaisesRegex(ValueError,'strong rolling main'):self.run_audit()

    def test_cond_report_cannot_be_deleted(self):
        self.pickup_registration()
        with self.assertRaisesRegex(ValueError,'cond_hist report'):self.run_audit()

    def test_old_program_cannot_replace_new_scope(self):
        self.reg['program']['sha256']=g.LEGACY_PROGRAM_SHA
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):self.run_audit()

    def test_missing_amendment_binding_cannot_pass(self):
        del self.reg['scope_amendment']
        with self.assertRaisesRegex(ValueError,'artifact must'):self.run_audit()

if __name__=='__main__':unittest.main()
