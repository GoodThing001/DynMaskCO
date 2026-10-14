"""Preserve NumPy-major cross-runtime identity failure; pin legacy runtime QA."""
import importlib.util,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'research/review_3080_20261010/pickup'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    source=ROOT/'tools/qualify_pickup_atomic_installation_root_v1.py'
    spec=importlib.util.spec_from_file_location('oldqa',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    _,dec=m.codecs();raw=dec(json.loads((m.OUT/'RESULT.json').read_text()));old=dec(json.loads(m.INPUT.read_text()))
    failures=[c['name'] for c in raw['checks'] if not c['pass']]
    assert len(failures)==5 and all(x.endswith(':actual_full_C0_certificate') for x in failures)
    target=P/'atomic_plan_installation_root_v2';target.mkdir(exist_ok=False)
    original=P/'atomic_plan_installation_root_v1/atomic_install.py';(target/'atomic_install.py').write_bytes(original.read_bytes())
    text=source.read_text().replace('atomic_plan_installation_root_v1','atomic_plan_installation_root_v2')
    text=text.replace("def worker():\n", "def worker():\n    import numpy as np\n    if np.__version__ != '1.26.4': raise RuntimeError('Pinned legacy NumPy1.26.4 engineering runtime required')\n")
    marker="    print(json.dumps({'kind':'ready','cpu':cpu,'source_start':start}),flush=True)"
    assert text.count(marker)==1
    text=text.replace(marker,"    runtime={'executable':sys.executable,'python':sys.version,'numpy':np.__version__,'scalar_promotion_example_dtype':str((np.float32(3)+3.).dtype)}\n    for path in (Path(sys.executable),Path(np.__file__),Path(np.core._multiarray_umath.__file__)):\n        start[str(path)]=sha(path)\n    print(json.dumps({'kind':'ready','cpu':cpu,'source_start':start,'runtime':runtime}),flush=True)")
    text=text.replace("    affinity();encode,decode=codecs();import public_fleet as pf", "    import numpy as np\n    if np.__version__ != '1.26.4':raise RuntimeError('Actual parent NumPy1.26.4 required')\n    affinity();encode,decode=codecs();import public_fleet as pf")
    text=text.replace("'input_sha256':sha(INPUT)", "'required_numpy':'1.26.4','actual_python':sys.version,'actual_executable':sys.executable,'executable_sha256':sha(sys.executable),'input_sha256':sha(INPUT)")
    new=ROOT/'tools/qualify_pickup_atomic_installation_root_v2.py'
    with new.open('x') as f:f.write(text)
    drifts=[]
    for r,o in zip([x for x in raw['records'] if 'mode' in x],old['records']):
        a=r['prepared']['certificate']['evaluation'];b=o['actual']['selected_certificate']['evaluation']
        drifts.append({'mode':r['mode'],'energy_diff_kwh':a['energy_kwh']-b['energy_kwh'],'tomato_quality_diff':a['tomato_mean_quality']-b['tomato_mean_quality'],'utility_diff':a['utility']-b['utility'],'same_hard_feasible':a['hard_feasible']==b['hard_feasible']})
    receipt={'failure':'NumPy2.5.3 engineering invocation did not exactly reproduce NumPy1.26.4 legacy certificate values. All five plan-pointer operations passed but full component identity qualification remains false in v1.',
        'v1_raw_sha256':sha(m.OUT/'RESULT.json'),'v1_source_sha256':sha(source),'new_source_sha256':sha(new),'installer_byte_identical':sha(original)==sha(target/'atomic_install.py'),
        'old_products_preserved':True,'drifts':drifts,'repair':'Use actual original prototype NumPy1.26.4 runtime; pin interpreter/NumPy/native module identity; no tolerance relaxation, policy or physical-source modification.',
        'reference':'https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html',
        'promotion_explanation':'Documented NumPy2 scalar promotion is a candidate cause, not yet a proven isolated numerical cause for this trace.',
        'new_scientific_data':False,'stable_online_gain':False}
    with (m.OUT/'RUNTIME_IDENTITY_FAILURE.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps({'old_failure_preserved':True,'new_launcher':str(new),'tolerance_relaxed':False}))
if __name__=='__main__':main()
