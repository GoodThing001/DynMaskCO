"""Verify numerical identities and preserved historical inputs, without rerouting."""
import ast
import hashlib
import json
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from decimal import Decimal as D
from pathlib import Path

sys.path.insert(0, str(next(p / 'scripts' for p in Path(__file__).resolve().parents
                           if (p / 'scripts/project_paths.py').is_file())))
from project_paths import EXTENSION_ROOT

ROOT = EXTENSION_ROOT
DOC = ROOT/'docs/static_delivery_research_20261020/direct_value_evidence'
ADAPTER = ROOT/'comparisons/static_coldchain/adapters'
sys.path.insert(0, str(ADAPTER))
import test_direct_value_evidence


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result_file = ROOT/'results/static_delivery/direct_value_evidence_20261020.json'
    result = json.loads(result_file.read_text(encoding='utf-8'))
    cfg = ROOT/'configs/static_delivery/direct_value_evidence_v2.json'
    assert result['config_sha256'] == sha(cfg)
    assert result['script_sha256'] == sha(ADAPTER/'direct_value_evidence.py')
    kg = D(25)*80*D('0.45359237')
    assert kg == D('907.18474000')
    rows = result['source_specific_value_quotes']
    assert len(rows) == 9
    for row in rows:
        assert abs(D(str(row['pallet_kg_derived']))-kg) < D('1e-9')
        high = D(str(row['high_quality_USD_per_pallet']))
        ratio = D(str(row['normalized_low_value']))
        assert abs(D(str(row['low_quality_USD_per_kg']))-high*ratio/kg) < D('1e-12')
        assert not row['quality_from_routes_inferred']
    thermal = result['v1_thermal_cost_unit_check']
    assert D('0.00005')*1000*60 == D(str(thermal['EUR_per_thermal_kWh']))
    assert D(1728)*60*D('0.00005') == D(str(thermal['EUR_per_vehicle_hour']))
    assert not result['new_profit_result'] and not result['new_routing_optimization_run']
    source = DOC/'sources/south_india_tomato_2022.xml'
    assert sha(source) == '659188ef29a45b1e9bfb6224393a7a3f3e488f057b4293a20a00b1418d9f8da7'
    tree = ET.parse(source)
    doi = [x.text for x in tree.findall('.//article-id') if x.attrib.get('pub-id-type') == 'doi']
    assert '10.1093/ajcn/nqac039' in doi
    text = ' '.join(tree.getroot().itertext()).lower()
    assert 'expected' in text and 'request' in text
    preserved = {
        'configs/static_delivery/product_evidence_v1.json': '3dd8c10768781145fe3a084dbce2c2e80cb7e13351de89d17ef6cab646b53947',
        'results/static_delivery/net_benefit_scenario_pilot_20261020/results.json': '02f3b2ee2f67547501afed1ef3f8514cfe567992062294cd676d94830d39391b',
        'results/static_delivery/net_benefit_scenario_pilot_20261020/results_metadata_corrected.json': 'd8a5669960af14eabaa20178e3f007c1210b10c06e254dfd51a24af6c129506a',
    }
    for name, expected in preserved.items():
        assert sha(ROOT/name) == expected, name
    old = ROOT/'results/static_delivery/net_benefit_scenario_pilot_20261020/scenario_config.json'
    assert old.read_bytes() == (ROOT/'configs/static_delivery/net_benefit_scenario_v1.json').read_bytes()
    with (DOC/'tests.log').open('w', encoding='utf-8') as log:
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_direct_value_evidence)
        checked = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    assert checked.wasSuccessful() and checked.testsRun == 6
    native = [ADAPTER/'direct_value_evidence.py', ADAPTER/'test_direct_value_evidence.py', Path(__file__), ROOT/'tools/evidence/acquire/acquire_direct_value_evidence_20261020.py']
    for p in native:
        ast.parse(p.read_text(encoding='utf-8'))
    assert not (ADAPTER/'__init__.py').exists()
    legacy_diff = subprocess.run(['git', 'diff', '--numstat', 'HEAD', '--', 'MASKCO_code', 'CC_Compare'], cwd=ROOT.parent, capture_output=True, check=True, text=True)
    assert not legacy_diff.stdout.strip()
    docs = [DOC/'README.md', ROOT/'项目当前状态.md', ROOT/'docs/static_delivery_research_20261020/README.md', ROOT/'docs/文档索引与版本状态.md', ROOT/'docs/static_delivery_research_20261020/导师汇报摘要.md', ROOT/'docs/static_delivery_research_20261020/net_benefit_scenario/README.md', ROOT/'comparisons/static_coldchain/README.md']
    local_links = 0
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            resolved = p.parent/target.split('#')[0]
            # verification.json is this script's output, created after these checks.
            assert resolved.exists() or resolved.resolve() == (DOC/'verification.json').resolve(), (p, target)
            local_links += 1
    output = {'status': 'SOURCE_VALUE_QUOTES_AND_UNIT_GUARDS_VERIFIED_INCOMPLETE_ROUTING_PROFILE', 'semantic_tests_passed': checked.testsRun, 'independent_decimal_quote_checks': len(rows), 'south_india_XML_sha256': sha(source), 'new_config_sha256': sha(cfg), 'new_calculations_sha256': sha(result_file), 'historical_files_SHA_unchanged': preserved, 'historical_net_benefit_config_matches_saved_run': True, 'legacy_upstream_diff_empty': True, 'native_syntax_checks': len(native), 'local_links_checked': local_links, 'Gaukler_access': 'indexed primary passages only; directPDF403; no local fulltext or page visual check', 'complete_quality_mapping_obtained': False, 'new_routing_or_training': False, 'new_profit_result': False, 'git_commit_or_push': False}
    (DOC/'verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
