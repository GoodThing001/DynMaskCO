"""Freeze unfinished executor deliveries as root-reviewed, isolated candidates."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WT=Path(r'C:\Users\mmc\.codex\worktrees\ba63\MASKCO-Main\.arbor')
BASE=ROOT/'research/review_3080_20261010'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def copy(line,name):
    src=WT/line/'research/review_3080_20261010'/line/name
    dst=BASE/line/name
    dst.mkdir(exist_ok=False)
    rows=[]
    for p in src.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
        q=dst/p.relative_to(src);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        rows.append({'path':q.relative_to(dst).as_posix(),'source_sha256':sha(p),'copied_sha256':sha(q)})
    assert all(r['source_sha256']==r['copied_sha256'] for r in rows)
    (dst/'ROOT_INTEGRATION_IDENTITY.json').write_text(json.dumps({'exact_copy':True,'files':rows,
        'independent_model_execution':False,'formal_qualification':False},indent=2),encoding='utf-8')
    return dst
if __name__=='__main__':
    p=copy('pickup','actual_masked_fleet_followup_v1')
    b=copy('reproduction','formal_run_requirements_20261011')
    # Preserve executor v1. Correct the NEW adapter in a separate root v2.
    b2=BASE/'reproduction/formal_run_requirements_root_v2';shutil.copytree(b,b2)
    f=b2/'new_run_gate_adapter.py';s=f.read_text(encoding='utf-8')
    s=s.replace("{'cond_hist', 'same_init_reconstruction_control'}", "{'qualified_strong_nonlearned_rolling', 'same_init_reconstruction_control'}")
    s=s.replace("by_role['cond_hist']['minimum_effect']", "by_role['qualified_strong_nonlearned_rolling']['minimum_effect']")
    s=s.replace("reject('pickup main and learning attribution roles required')", "reject('pickup strong rolling main and learning attribution roles required')")
    anchor="        if by_role['qualified_strong_nonlearned_rolling']['minimum_effect'] < 20"
    s=s.replace(anchor,"        if 'cond_hist' not in arms:\n            reject('pickup cond_hist report arm must remain present')\n"+anchor)
    f.write_text(s,encoding='utf-8')
    f=b2/'test_new_run_gate.py';s=f.read_text(encoding='utf-8')
    s=s.replace("['cond_hist','same_init_reconstruction_control']", "['qualified_strong_nonlearned_rolling','same_init_reconstruction_control']")
    s=s.replace("self.reg['line']='pickup'", "self.reg['line']='pickup'\n        self.reg['arms'].append('cond_hist')")
    s=s.replace("if __name__=='__main__':unittest.main()", """class PickupMainRoleRegression(unittest.TestCase):
    def test_adapter_does_not_restore_old_cond_main(self):
        source=Path(g.__file__).read_text(encoding='utf-8')
        self.assertNotIn(\"by_role['cond_hist']['minimum_effect']\",source)
        self.assertIn(\"by_role['qualified_strong_nonlearned_rolling']['minimum_effect']\",source)

if __name__=='__main__':unittest.main()""")
    f.write_text(s,encoding='utf-8')
    (b2/'ROOT_PROTOCOL_CORRECTION.json').write_text(json.dumps({'reason':'v1 incorrectly restored historical cond_hist main; amended prospective main is strong_rolling >=20; cond_hist remains report arm',
        'old_adapter_sha256':sha(b/'new_run_gate_adapter.py'),'new_adapter_sha256':sha(b2/'new_run_gate_adapter.py'),
        'old_archive_changed':False,'formal_gain_established':False},indent=2),encoding='utf-8')
    print(json.dumps({'pickup':str(p),'reproduction_v1':str(b),'root_corrected_candidate':str(b2)}))
