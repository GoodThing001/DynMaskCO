"""Verify this delivery without re-running historical experiments or training."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(next(p / 'scripts' for p in Path(__file__).resolve().parents
                           if (p / 'scripts/project_paths.py').is_file())))
from project_paths import EXTENSION_ROOT, WORKSPACE_ROOT

E=EXTENSION_ROOT
B=E/'docs/static_delivery_research_20261020'
C=E/'comparisons/static_coldchain'
D=E/'data/classic_static_sources_20261020'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))

def main():
    source=E/'data/public_sources_20261020/refrigerated_routing_rrp'
    old=read(source/'download_manifest.json')
    assert len(old)==61 and all(sha(source/r['file'])==r['sha256'] for r in old)
    classic=read(D/'solomon_100/FILES_SHA256.json')
    assert len(classic)==56 and all(sha(D/'solomon_100/instances'/r['path'])==r['sha256'] for r in classic)
    author_root=D/'VRPTWQ_Solomon'
    inner=next(author_root.glob('*.zip'))
    with zipfile.ZipFile(inner) as z:
        members=[m for m in z.infolist() if not m.is_dir()]
        assert len(members)==56
        assert all((author_root/'instances'/m.filename).read_bytes()==z.read(m) for m in members)
    pins=read(C/'SOURCE_PINS.json')
    assert len(pins)==8
    upstream_files=0
    for pin in pins:
        folder=C/'upstream'/pin['name']
        provenance=read(folder/'SNAPSHOT_PROVENANCE.json')
        assert sha(C/'archives'/f"{pin['name']}-{pin['commit'][:12]}.zip")==pin['archive_sha256']
        assert all(sha(folder/r['path'])==r['sha256'] for r in provenance['files'])
        upstream_files+=len(provenance['files'])
    assert 'version = "0.14.0"' in (C/'upstream/PyVRP-0.14.0/pyproject.toml').read_text()
    result=read(E/'results/static_delivery/classic_solomon_preliminary_v2_20261020/results.json')
    assert result['script_sha256']==sha(C/'adapters/solomon_preliminary.py')
    assert len(result['rows'])==12 and all(r['report']['feasible'] for r in result['rows'])
    quality=read(E/'results/static_delivery/quality_evidence_demo_final_20261020.json')
    assert quality['config_sha256']==sha(E/'configs/static_delivery/product_evidence_v1.json')
    assert quality['script_sha256']==sha(C/'adapters/quality_evidence_demo.py')
    assert len(quality['rows'])==30 and all(r['sales_value'] is None for r in quality['rows'])
    native=list(C.glob('adapters/*.py'))+list((E/'tools/evidence').rglob('*static*20261020.py'))
    for p in native: ast.parse(p.read_text(encoding='utf-8'))
    assert not list((C/'adapters').rglob('__init__.py'))
    test_path=C/'adapters/test_evidence_and_solomon.py'
    spec=importlib.util.spec_from_file_location('static_research_tests',test_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    stream=io.StringIO()
    tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    (B/'tests_final.log').write_text(stream.getvalue(),encoding='utf-8')
    assert tests.wasSuccessful() and tests.testsRun==9
    local_links=[]
    docs=list(B.glob('*.md'))+list(C.glob('*.md'))+[D/'README.md']
    for p in docs:
        for url in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if '://' in url or url.startswith('#'): continue
            dest=url.split('#')[0]
            if p.name=='README.md' and p.parent==B and dest=='verification.json': continue
            target=(p.parent/dest).resolve()
            assert target.exists(),(str(p),dest)
            local_links.append({'document':str(p.relative_to(E)),'target':dest})
    diff=subprocess.run(['git','diff','--numstat','--','MASKCO_code','CC_Compare'],cwd=WORKSPACE_ROOT,capture_output=True,text=True,check=True)
    assert not diff.stdout.strip()
    source_files=[{'file':str(p.relative_to(E)),'sha256':sha(p),'bytes':p.stat().st_size} for p in (B/'sources').iterdir() if p.suffix in ('.pdf','.xml')]
    verification={
        'status':'RESEARCH_AND_PRELIMINARY_ROUTING_VERIFIED_NOT_NET_PROFIT',
        'original_RRP_files_SHA256_verified':61,'original_Solomon_files_SHA256_verified':56,
        'author_VRPTWQ_extracted_original_bytes_verified':56,
        'official_source_repositories':7,'source_snapshots':8,'source_snapshot_members_SHA256_verified':upstream_files,
        'PyVRP_runtime_matching_source_commit':'5d9776a954b810bdb3fe71d47b1e7de7cffe90d2',
        'older_PyVRP_snapshot_preserved':'4b2aa285f34efb8e2ff1995a87b834414c1c5c17; project version1.0.0a0; not the executed0.14.0 release',
        'routing_results_feasible':12,'independent_audit_tests':tests.testsRun,'failures':len(tests.failures),'errors':len(tests.errors),
        'quality_only_calculations':30,'net_profit_computed':False,'new_neural_training_executed':False,
        'method_gain_established':False,'legacy_upstream_diff_empty':True,
        'downloaded_PDF_and_XML':source_files,'source_entries':len(read(B/'sources.json')),
        'method_assessments':len(read(C/'method_eligibility.json')),
        'own_adapter_syntax_files':len(native),'own_package_init_added':False,
        'local_document_links_checked':len(local_links),
        'user_confirmed':{'fish':'chilled aerobic mullet slices','pricing_data':'none; keep gaps and deliver routing/quality preliminary results'},
        'vehicle_source_access':'2023 CO2 official PDF text extracted via web; direct HTTP403 and screenshot cache miss; no local PDF claimed',
        'git_commit_or_push_by_this_task':False,
        'limitations':['classic routing results not food net-benefit results','no same-product calibrated price-quality mapping','unknown initial quality/age and order-product mapping','chilled equipment capacity/duty-cycle and energy price still incomplete','one pilot seed per neural-free traditional method; no claim of fair full method comparison']}
    (B/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k not in ('downloaded_PDF_and_XML','limitations')},ensure_ascii=True))

if __name__=='__main__':main()
