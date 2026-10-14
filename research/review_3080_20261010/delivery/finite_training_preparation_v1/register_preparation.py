"""Register explicit engineering inputs and an unexecutable science draft."""
import argparse
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parent/'actual_checkpoint_engineering_v1'))
from finite_common import sha,dump
from actual_full_step_qa import sources


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--upstream',type=Path,required=True)
    a=ap.parse_args()
    for folder in ('delivery','coldchain'):
        sys.path.insert(0,str(a.root/'scripts'/folder))
    from delivery_contract import DeliveryContract
    old=json.loads((HERE.parent/'actual_checkpoint_engineering_v1/ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json').read_text(encoding='utf-8'))
    common={'mode':'pre_dispatch_snapshot','contract_hash':DeliveryContract().identity(),
        'contract_payload':DeliveryContract().payload(),'dependency_sources':sources(a.root,a.upstream),
        'original_checkpoint_sha256':old['original_checkpoint_sha256'],'original_model_config':old['model_config'],
        'programs':{n:sha(HERE/n) for n in ('finite_common.py','teacher_pipeline.py','train_full_masked.py')},
        'sibling_programs':{n:sha(HERE.parent/n) for n in ('features_objective_qa_v1/features_objective.py',
              'actual_checkpoint_engineering_v1/inspect_checkpoint.py')},
        'teacher_solver':'single fixed masked distance-regret2 candidate perstate; no logits; full C0 certification',
        'teacher_budget_each_instance_s':10.,'teacher_masks_each':4,'teacher_max_each_mask_s':2.5,
        'generation_included_in_teacher_budget':True,'mask_fraction':.5,
        'teacher_CE_target':'min certified full C0 pair cost; tie incumbent; invalid/late incumbent',
        'teacher_preference':'candidate vs incumbent original order; sign(Cinc-Ccandidate); absdelta<=1e-8 gives0',
        'invalid_negative_small_states':'retain all; invalidpref0; noresample/no target afterbudget success',
        'optimizer':{'name':'sgd_without_momentum','learning_rate':.00001},'preference_coefficient':.2,
        'checkpoint_selection':'fixed_final_step','dynamic_online_claim_enabled':False,
        'physical_calibration_claim_enabled':False,'training_parameters_are_new_project_choices':True}
    engineering=dict(common,version='D-FULLMASK-FINITE-ENGINEERING-v1',status='REGISTERED_ENGINEERING_ONLY',
        engineering_seed=202610110071,engineering_mask_seed=202610110072,engineering_schedule_seed=202610110073,
        engineering_instance_ids=[900000,900001],engineering_training_steps_each=2,engineering_batch_size=4,
        research_model_eligible=False,full_training_outer_budget_seconds=300)
    dump(HERE/'FINITE_ENGINEERING_REGISTRATION.json',engineering)
    draft=dict(common,version='D-FULLMASK-SNAPSHOT-TRAIN-v1',status='DRAFT',train_instances=64,masks_per_instance=4,
        train_instance_ids=None,training_steps=128,batch_size=4,development_instances=16,
        confirmation_instances_each=40,confirmation_cohorts=3,seeds=None,
        accepted_gate_sha256={'teacher_engineering':None,'actual_10D_objective':None,'deployment_10s_budget':None},
        deployment_budget_s=10.,deployment_qualification='actual total includesfeatures,inference,search,certification,submit',
        deployment_arms=['paired_objective','reconstruction_control','original_warmstart','random10D_untrained','ANY50','GLS'],
        deployment_trial_order='registered balanced randomized arm order',
        development_scope='16new instances diagnostics only; no checkpoint or hyperparameter selection',
        confirmation_scope='3fresh cohorts x40; never old DEV; only after training and budget acceptance',
        confirmation_primary='paired_objective relative reduction vsANY50 andGLS mean>=.01 andpairedbootstrap2000 CI lower>0 inall3cohorts',
        attribution='paired_objective vsCEOnly mean>0 and95CI lower>0 inall3cohorts',
        bootstrap_method='fixed registered2000 percohort paired instance resamples; no pooling or seed selection',
        baseline_scope='GLS proposesdistance, fullC0cert/selection; not C0optimality; actualnewrawroutes/manifestrequired',
        pretraining_attribution_gap='untrainedrandom doesnotisolate pretraining; matchedrandom-trained full128 control separatelyrequiredforclaim',
        holdouts_prohibited=[20260927,20260928],research_model_eligible=False,
        start_barriers=['teacherengineeringaccepted','actual10Dfullobjectivesaccepted',
            'actualallarms10scompletebudgetintegrationaccepted','beforefirst64generationnewregisteredsciencejson',
            'freshsplitidentityhistoryreview','sourcecheckpointinitialendmanifest'])
    dump(HERE/'FULL_MASKED_FINITE_PROTOCOL_DRAFT.json',draft)
    print(json.dumps({'engineering_sha256':sha(HERE/'FINITE_ENGINEERING_REGISTRATION.json'),
                     'draft_sha256':sha(HERE/'FULL_MASKED_FINITE_PROTOCOL_DRAFT.json')}))
