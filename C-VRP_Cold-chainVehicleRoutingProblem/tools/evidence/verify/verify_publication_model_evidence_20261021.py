"""Record bounded paper-model evidence and verify source bytes and unit conversions.

No customer instances, route results, calibrated product parameters or rankings are
generated. Journal labels distinguish a dated public table from undated reports.
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import fitz
import xlrd

sys.path.insert(0, str(next(p / 'scripts' for p in Path(__file__).resolve().parents
                           if (p / 'scripts/project_paths.py').is_file())))
from project_paths import EXTENSION_ROOT

ROOT = EXTENSION_ROOT
OUT = ROOT / 'docs/static_delivery_research_20261021'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n', encoding='utf-8')


def main():
    pdf_specs = [
        ('sources/joint_distribution_RCR_2020.pdf',
         '8cfed39271a1fa93afef50f1f79684e4e89b1a6237d3209bb20a50a9d5b5f878',
         '10.1016/j.resconrec.2020.104715', 30),
        ('sources/precooling_EJOR_2025.pdf',
         '92a1632e81ff4106deb72230961b5574f021e69ef5c0ca9582783a5627805988',
         '10.1016/j.ejor.2024.09.041', 21),
    ]
    pdf_rows = []
    for name, expected, doi, pages in pdf_specs:
        path = OUT / name
        assert path.read_bytes().startswith(b'%PDF-') and sha(path) == expected
        with fitz.open(path) as doc:
            assert len(doc) == pages
            contents = '\n'.join(p.get_text() for p in doc)
            assert doi in re.sub(r'\s+', '', contents)
        pdf_rows.append(dict(path=name, sha256=expected, doi=doi, pages=pages))

    table = OUT / 'sources/CAS2025_Q1_SCNU.xls'
    assert sha(table) == '0730277ebe52b6c3114d811dead97abeb30d0664e2a3351f302649f098ef6067'
    book = xlrd.open_workbook(str(table))
    cas_rows = []
    for sheet, row, issn, journal in [
        ('环境科学与生态学', 19, '0921-3449', 'RESOURCES CONSERVATION AND RECYCLING'),
        ('工程技术', 62, '0925-5273', 'INTERNATIONAL JOURNAL OF PRODUCTION ECONOMICS'),
    ]:
        values = book.sheet_by_name(sheet).row_values(row - 1)
        assert values[1:] == [journal, issn, '1区']
        cas_rows.append(dict(sheet=sheet, excel_row=row, values=values))

    institution_jcr = dict(
        label='Q1', edition=None, category=None,
        evidence_level='institutional_report_NOT_Clarivate_current_export')
    models = [
        dict(id='liu2020', doi=pdf_specs[0][2], journal='Resources, Conservation and Recycling',
             year=2020, source_url='https://pure.coventry.ac.uk/ws/files/28060196/Binder6.pdf',
             source=pdf_rows[0], locations=['Eq.3-8', 'Table5 (printed p16, PDF p17)'],
             product='same frozen food; multi-depot joint-distribution case',
             travel_loss='P*m*(1-epsilon1*exp(-theta*(arrival-departure)))',
             unload_loss='P*remaining_after_unload*(1-epsilon2*exp(-theta*m/unload_rate))',
             printed_parameters=dict(price_RMB_per_t=5000, distance_RMB_per_km=3,
                 fuel_RMB_per_L=6.68, cold_travel_L_per_h=2, cold_unload_L_per_h=2.5,
                 epsilon1=1, epsilon2=0.9, theta=0.002, theta_printed_unit='none',
                 speed_km_per_h=40, unload_t_per_h=3.6, payload_t=2,
                 fixed_bundle_RMB_per_vehicle=200),
             clarification='Hour interpretation requires theta in h^-1; not a unit explicitly printed by source. Fixed bundle includes wages. Epsilon2 is outside exponential.',
             CAS=dict(edition=2025, label='1区', evidence=cas_rows[0]),
             JCR=dict(**institution_jcr, url='https://clxy.bjut.edu.cn/info/1095/1736.htm')),
        dict(id='stellingwerf2021', doi='10.1016/j.ijpe.2020.107849',
             journal='International Journal of Production Economics', year=2021,
             source_path='../static_delivery_research_20261020/sources/quality_driven_2021.pdf',
             source_url='https://transsonic.nl/documents/Stellingwerf_EtAl_IJPE_2021_The_Quality_Driven_Vehicle_Routing_Problem.pdf',
             model='zero-order temperature-dependent relative quality loss',
             units=dict(kinetic_time='second', cost='EUR', temperature='K'),
             quality_threshold=0.8, location='Section4.3.1 and Table5 (p8)',
             case_specific_limits=dict(infeasible_at_or_above=0.85, nonbinding_below=0.71),
             limitation='These feasibility limits are not universal food standards or a monetary price function.',
             CAS=dict(edition=2025, label='1区', evidence=cas_rows[1]),
             JCR=dict(**institution_jcr, url='https://som.xjtu.edu.cn/info/1126/13000.htm')),
        dict(id='lin2025', doi=pdf_specs[1][2], journal='European Journal of Operational Research',
             year=2025, source_url='https://edepot.wur.nl/675716', source=pdf_rows[1],
             quality='exp(-integral(alpha(T(s)),s=harvest_time..evaluation_time)); batch quality averages over harvest interval',
             alpha='0.0048*exp(0.1036*T)', product='sweet corn example fitted from Blackburn and Scudder (2009)',
             units=dict(time='hour', temperature='degree_Celsius', cost='RMB'),
             main_thresholds=[0.90, 0.91, 0.92, 0.93], main_location='Section5.1.1, PDF p11',
             temperature_scenarios=dict(low=0.95, high=0.8, location='Table2, PDF p15'),
             limitation='Precooling pickup problem, not static delivery; no calibrated monetary value curve.',
             CAS=dict(edition=None, label='2区', evidence_level='institutional_report'),
             JCR=dict(**institution_jcr, url='https://bas.gzhu.edu.cn/info/1069/2913.htm')),
        dict(id='huang2024', doi='10.1016/j.cie.2024.110022',
             journal='Computers & Industrial Engineering', year=2024,
             evidence_level='primary_publisher_index_NOT_full_parameter_tables',
             source_url='https://doi.org/10.1016/j.cie.2024.110022',
             use='Recent ALNS and quality/satisfaction constrained routing reference; exact parameters not asserted.',
             CAS=dict(edition=None, label=None, status='conflicting_1_and_2_institutional_labels',
                 urls=['https://bas.gzhu.edu.cn/info/1069/2913.htm',
                       'https://facultyce.szu.edu.cn/dingzhikun/zh_CN/kyqklw/5371/content/8692.htm']),
             JCR=dict(**institution_jcr, url='https://som.xjtu.edu.cn/info/1126/13000.htm')),
    ]
    save('model_evidence.json', dict(date='2026-10-21', scope='bounded model-selection evidence, not a systematic review',
         CAS2025_table=dict(url='https://statics.scnu.edu.cn/pics/lib/2025/1031/1761900158678161.xls',
                            sha256=sha(table), verified_rows=cas_rows),
         ranking_limit='Current 2026 Clarivate/CAS authenticated edition/category NOT certified; no SJR substitution.',
         models=models, execution_status='research_protocol_only_no_new_training_no_new_routing_results'))

    converted = dict(price_CNY_per_kg=5000/1000, distance_CNY_per_km=3,
        cold_travel_CNY_per_h=6.68*2, cold_unload_CNY_per_h=6.68*2.5,
        cold_travel_CNY_per_min=6.68*2/60, cold_unload_CNY_per_min=6.68*2.5/60,
        unloading_kg_per_min=3.6*1000/60, theta_per_min_if_hour_interpretation=0.002/60,
        travel_quality_at_24h=math.exp(-0.002*24),
        hours_to_80pct_travel_quality=-math.log(0.8)/0.002,
        original_unload_loss_fraction_at_zero_duration=1-0.9*math.exp(0))
    assert math.isclose(converted['cold_travel_CNY_per_min']*60, 13.36)
    assert math.isclose(converted['cold_unload_CNY_per_min']*60, 16.7)
    assert math.isclose(math.exp(-(0.002/60)*1440), converted['travel_quality_at_24h'])
    assert converted['travel_quality_at_24h'] > 0.95
    assert converted['hours_to_80pct_travel_quality'] > 111
    assert math.isclose(converted['original_unload_loss_fraction_at_zero_duration'], 0.1)
    page_rows = [dict(path=str(p.relative_to(OUT)), sha256=sha(p), visually_inspected=True)
                 for p in sorted((OUT/'page_checks').glob('*.png'))]
    assert len(page_rows) == 6
    preserved = []
    for name, expected in [
        ('configs/static_delivery/product_evidence_v1.json', '3dd8c10768781145fe3a084dbce2c2e80cb7e13351de89d17ef6cab646b53947'),
        ('results/static_delivery/net_benefit_scenario_pilot_20261020/results.json', '02f3b2ee2f67547501afed1ef3f8514cfe567992062294cd676d94830d39391b'),
        ('results/static_delivery/net_benefit_scenario_pilot_20261020/results_metadata_corrected.json', 'd8a5669960af14eabaa20178e3f007c1210b10c06e254dfd51a24af6c129506a'),
    ]:
        assert sha(ROOT/name) == expected
        preserved.append(dict(path=name, sha256=expected))
    report = OUT/'论文实验协议.md'
    local_links = []
    # Materialize the report's self-linked verification artifact before link checks.
    save('verification.json', dict(status='PENDING_LOCAL_LINK_CHECK'))
    for target in re.findall(r'\]\(([^)]+)\)', report.read_text(encoding='utf-8')):
        if '://' not in target:
            path = report.parent/target
            assert path.exists(), target
            local_links.append(target)
    save('verification.json', dict(status='PASS', date='2026-10-21', source_pdfs=pdf_rows,
         CAS_rows=cas_rows, page_checks=page_rows, conversions=converted,
         conversions_identity_checked=True, preserved_files=preserved, local_links=local_links,
         no_new_training=True, no_new_route_results=True))
    print(json.dumps(dict(status='PASS', source_pdfs=len(pdf_rows), CAS_rows=len(cas_rows),
                         inspected_pages=len(page_rows), preserved_files=len(preserved),
                         conversions=converted), ensure_ascii=True))


if __name__ == '__main__':
    main()
