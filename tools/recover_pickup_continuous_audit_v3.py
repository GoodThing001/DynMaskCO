"""Preserve pre-audit bootstrap failure; new source, identical actual inputs."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
T=Path(__file__).resolve().parent;O=T.parent/'research/review_3080_20261010/pickup/continuous_checkpoint_root_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    src=T/'audit_pickup_continuous_checkpoint_v2.py';dest=T/'audit_pickup_continuous_checkpoint_v3.py';shutil.copy2(src,O/'FROZEN_audit_pickup_continuous_checkpoint_v2.py')
    text=src.read_text(encoding='utf-8').replace('import hard_admission,rolling_optimizer\n','').replace('import qualify_pickup_joint_policy_transaction_root_v3 as old','import qualify_pickup_joint_policy_transaction_root_v3 as old\nimport hard_admission,rolling_optimizer')
    with dest.open('x',encoding='utf-8') as f:f.write(text)
    reg=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='Original independent v2 import bootstrap failure before audit registration/computation. Preserve actual exit1. v3 only moves existing hard_admission/rolling_optimizer imports after original path bootstrap. Same actual source/runtime/raw payload input bytes, no producer rerun.',actual_original_exit_code=1,original_error="ModuleNotFoundError: No module named 'hard_admission'",old_source_sha256=sha(src),new_source_sha256=sha(dest),production_registration_sha256=sha(O/'REGISTRATION.json'),producer_result_sha256=sha(O/'RESULT.json'),new_audit_source=str(dest))
    with (O/'AUDIT_BOOTSTRAP_RECOVERY_REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(reg,f,indent=2)
    print('new audit loader registered; producer and failed audit source untouched')
if __name__=='__main__':main()
