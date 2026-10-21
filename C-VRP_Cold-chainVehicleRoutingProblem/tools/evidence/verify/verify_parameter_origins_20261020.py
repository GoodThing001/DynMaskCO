"""Verify research provenance and frozen results; does not solve or generate cases."""
import ast
import hashlib
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(next(p / 'scripts' for p in Path(__file__).resolve().parents
                           if (p / 'scripts/project_paths.py').is_file())))
from project_paths import EXTENSION_ROOT, WORKSPACE_ROOT

E = EXTENSION_ROOT
B = E / 'docs/static_delivery_research_20261020'
F = B / 'parameter_origins_followup'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    downloads = read(F / 'acquisition_manifest.json')
    assert len(downloads) == 3
    for row in downloads:
        p = E / row['path']
        assert row['status'] == 'DOWNLOADED' and p.exists()
        assert sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
    candidates = read(F / 'parameter_candidates.json')
    assert not candidates['activated_in_solver']
    assert not candidates['customer_instances_generated']
    assert not candidates['profit_results_produced']
    by_id = {c['id']: c for c in candidates['candidates']}
    assert len(by_id) == 8
    econ = by_id['tomato_paper_economic_scenario']
    assert math.isclose(econ['purchase_price_per_kg_derived'], econ['selling_price_per_kg'] * econ['purchase_fraction_of_selling_price'])
    market = ET.parse(F / 'sources/mullet_market_2022.xml').getroot()
    table = next(t for t in market.findall('.//table-wrap') if t.attrib.get('id') == 'tbl5')
    row = next(r for r in table.findall('.//tr') if 'Mugil cephalus' in ''.join(r.itertext()))
    cells = [''.join(c.itertext()).strip() for c in row]
    vals = [float(v) for v in cells[1:6]]
    assert vals == list(by_id['mullet_historical_channel_prices']['prices_per_kg'].values())
    quality = read(E / 'results/static_delivery/quality_evidence_demo_final_20261020.json')
    config = E / 'configs/static_delivery/product_evidence_v1.json'
    assert quality['config_sha256'] == sha(config)
    assert quality['script_sha256'] == sha(E / 'comparisons/static_coldchain/adapters/quality_evidence_demo.py')
    routing = read(E / 'results/static_delivery/classic_solomon_preliminary_v2_20261020/results.json')
    assert routing['script_sha256'] == sha(E / 'comparisons/static_coldchain/adapters/solomon_preliminary.py')
    assert len(routing['rows']) == 12 and all(r['report']['feasible'] for r in routing['rows'])
    links = []
    for p in [F / 'README.md', B / 'README.md', B / '论文参数与价值关系.md', B / '导师汇报摘要.md']:
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            dest = (p.parent / target.split('#')[0]).resolve()
            if dest == F / 'verification.json':
                continue
            assert dest.exists(), (p.name, target)
            links.append({'document': str(p.relative_to(E)), 'target': target})
    images = list((F / 'page_checks').glob('*.png'))
    assert len(images) == 3
    for p in images:
        assert p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    native = [E / 'tools/evidence/acquire/acquire_parameter_origins_20261020.py', Path(__file__).resolve()]
    for p in native:
        ast.parse(p.read_text(encoding='utf-8'))
    diff = subprocess.run(['git', 'diff', '--numstat', '--', 'MASKCO_code', 'CC_Compare'], cwd=WORKSPACE_ROOT, capture_output=True, text=True, check=True)
    assert not diff.stdout.strip()
    report = {
        'status': 'PARAMETER_ORIGIN_RESEARCH_VERIFIED_NO_NEW_PROFIT_RUN',
        'downloaded_originals_SHA256_verified': len(downloads),
        'candidate_rows': len(by_id),
        'market_Table5_species_row_checked_against_original_XML': vals,
        'visually_reviewed_original_pages': [{'path': str(p.relative_to(E)), 'sha256': sha(p)} for p in images],
        'local_document_links_checked': len(links),
        'historical_quality_config_and_adapter_SHA_unchanged': True,
        'historical_routing_adapter_SHA_unchanged': True,
        'historical_quality_config_sha256': sha(config),
        'historical_quality_result_sha256': sha(E / 'results/static_delivery/quality_evidence_demo_final_20261020.json'),
        'historical_routing_result_sha256': sha(E / 'results/static_delivery/classic_solomon_preliminary_v2_20261020/results.json'),
        'legacy_upstream_diff_empty': True,
        'native_scripts_syntax_checked': len(native),
        'new_instances_generated': False, 'new_solves_or_training': False, 'profit_computed': False,
        'limits': ['publisher tomato economics accessed through search full-text; direct DOI403; no local PDF', 'three listed supplementary ZIPs not downloaded', 'graph initial values approximate, not exact fitted intercepts or current-batch measurements', 'generic author quality400 is not food-specific', 'historical v1 score9 flag superseded by this follow-up; frozen original preserved']
    }
    (F / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
