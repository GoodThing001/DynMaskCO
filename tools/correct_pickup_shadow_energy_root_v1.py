"""Instrument archived public states and create an isolated conserved-energy SAA."""
import ast,hashlib,inspect,json,sys
from dataclasses import is_dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';P=ROOT/'research/review_3080_20261010/pickup'
OUT=P/'shadow_energy_correction_root_v1'
sys.path.insert(0,str(P));import public_fleet as pf
pf.bootstrap(E/'scripts')
import scenario_saa,coldchain_contract,coldchain_state,strict_online_env
from audit_io import decode
from recourse_snapshot import restore_recourse_snapshot
CLASSES={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env) for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    OUT.mkdir(exist_ok=False);source=Path(scenario_saa.__file__)
    rawfile=P/'actual_masked_fleet_followup_v1/full_day_qa_v1/own_uncond_dynamic_fullraw.json'
    body=inspect.getsource(scenario_saa.SaaReplanner._sim_scenario)
    marker="        for s in st.values():\n            for o in s['route']:"
    if body.count(marker)!=1:raise ValueError('Original method shape changed; no broad rewriting')
    body=body.replace(marker,"        est_used += self._fixed_unavailable_energy\n"+marker)
    program='''"""Scoped resource-ledger correction; old SAA source/results remain unchanged."""
import copy, math
from scenario_saa import *
from scenario_saa import SaaReplanner as OriginalSAA
class EnergyCompleteSAA(OriginalSAA):
    def _sim_states(self,env,inst_idx,clock,vehicles,served_mask,plan):
        states=super()._sim_states(env,inst_idx,clock,vehicles,served_mask,plan)
        fixed_energy=0.;fixed_distance=0.;entries=[]
        for v in vehicles:
            if v.vehicle_id in states or v.coldchain_state is None:continue
            state=copy.deepcopy(v.coldchain_state)
            if v.status=='returning' and not state.closed:
                finish=float(v.return_finish)
                if not math.isfinite(finish) or finish<float(clock)-1e-6:
                    raise ValueError('Invalid immutable return deadline')
                distance=float(env.dist_mat[inst_idx,v.current_node,0])
                state,_=transition_segment(state,depart_time=float(clock),arrival_time=finish,service_finish=finish,
                    served_customer=None,active_zone_mask=(True,)*len(self.contract.thermal.supported_temp_classes),
                    contract=self.contract,return_to_depot=True,segment_distance_units=distance)
                fixed_distance+=distance
            elif not state.closed:
                raise ValueError('Excluded vehicle has no immutable closed/returning accounting')
            fixed_energy+=float(state.cumulative_energy_kwh)
            entries.append({'vehicle_id':int(v.vehicle_id),'status':v.status,'fixed_energy_kwh':float(state.cumulative_energy_kwh)})
        self._fixed_unavailable_energy=fixed_energy
        self._fixed_unavailable_distance=fixed_distance
        self.unavailable_ledger=entries
        return states
    def _complete_u(self,space,env,inst_idx,clock,st,wait_until):
        energy,distance=super()._complete_u(space,env,inst_idx,clock,st,wait_until)
        if energy is None:return None,None
        return energy+self._fixed_unavailable_energy,distance+self._fixed_unavailable_distance
'''+body
    ast.parse(program)
    (OUT/'energy_complete_saa.py').write_text(program,encoding='utf-8')
    write(OUT/'REGISTRATION.json',{'scope':'preserve old source; ledger correction QA only before any new online research',
        'original_SAA_source_sha256':sha(source),'new_adapter_sha256':sha(OUT/'energy_complete_saa.py'),
        'observed_full_day_raw_sha256':sha(rawfile),'past_and_immutable_return_energy_accounted':True,
        'all_arms_same_future_scoring_required':True,'S3_overage_price_unchanged_null':True,'formal_online_enabled':False})
    sys.path.insert(0,str(OUT));from energy_complete_saa import EnergyCompleteSAA
    raw=decode(json.loads(rawfile.read_text(encoding='utf-8')),CLASSES)
    rows=[];checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    for event in raw['events']:
        packet=event['public_packet'];vehicles,_,served=restore_recourse_snapshot(packet['snapshot'])
        common=dict(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],
            contract=packet['contract'],cooling_share=0.,sampler=None,reject_penalty={0:5.,1:10.,2:15.},energy_pricing='marginal')
        old=scenario_saa.SaaReplanner(**common);new=EnergyCompleteSAA(**common)
        env=strict_online_env.StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],len(vehicles),old,packet['contract'],packet['booking_horizon'])
        plan={int(k):v for k,v in packet['plan'].items()};clock=float(packet['snapshot']['clock'])
        states=old._sim_states(env,0,clock,vehicles,served,plan)
        corrected=new._sim_states(env,0,clock,vehicles,served,plan)
        all_observed=sum(v.coldchain_state.cumulative_energy_kwh for v in vehicles if v.coldchain_state is not None)
        active_observed=sum(s['cc'].cumulative_energy_kwh for s in states.values() if s['cc'] is not None)
        omitted=all_observed-active_observed
        closed=sum(v.coldchain_state.cumulative_energy_kwh for v in vehicles if v.vehicle_id not in states and v.coldchain_state is not None and v.coldchain_state.closed)
        ck(str(event['event_id'])+':active_action_set_unchanged',set(states)==set(corrected))
        ck(str(event['event_id'])+':fixed_reserve_covers_observed_excluded',new._fixed_unavailable_energy>=omitted-1e-8)
        ck(str(event['event_id'])+':closed_not_double_counted',new._fixed_unavailable_energy>=closed-1e-8)
        rows.append({'event_id':int(event['event_id']),'clock':clock,'observed_all_fleet_kwh':float(all_observed),
            'old_shadow_active_kwh':float(active_observed),'omitted_already_consumed_kwh':float(omitted),
            'new_fixed_unavailable_kwh':new._fixed_unavailable_energy,'immutable_return_distance_units':new._fixed_unavailable_distance,
            'unavailable_ledger':new.unavailable_ledger})
    ck('original_SAA_bytes_preserved',sha(source)==json.loads((OUT/'REGISTRATION.json').read_text())['original_SAA_source_sha256'])
    bad=[r for r in rows if r['omitted_already_consumed_kwh']>1e-8]
    report={'checks':checks,'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),'events':rows,
        'actual_archived_events_with_omitted_energy':len(bad),'max_omitted_observed_energy_kwh':max(r['omitted_already_consumed_kwh'] for r in rows),
        'bug_confirmed_in_observed_day':bool(bad),'old_numeric_results_rewritten':False,
        'new_scientific_online_run':False,'full_shadow_score_and_hard_deadline_regression_pending':True,'stable_core_online_gain':False}
    write(OUT/'ACTUAL_LEDGER_CORRECTION_QA.json',report)
    print(json.dumps({k:report[k] for k in ('checks_count','failed','actual_archived_events_with_omitted_energy','max_omitted_observed_energy_kwh')}))
    return int(report['failed']>0)
if __name__=='__main__':sys.exit(main())
