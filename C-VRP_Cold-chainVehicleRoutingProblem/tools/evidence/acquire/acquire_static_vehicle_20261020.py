import json
from acquire_static_evidence_20261020 import download, BASE

try:
    row=download(('chilled_CO2_unit_2023',
        'https://dwwl.ustc.edu.cn/ch/reader/create_pdf.aspx?file_no=202301009',
        BASE/'sources/chilled_CO2_unit_2023.pdf'))
except Exception as error:
    row={'name':'chilled_CO2_unit_2023','status':'FAILED','error_type':type(error).__name__,
         'error':str(error)[:240], 'official_PDF_text_verified_via_web':True,
         'local_PDF_acquired':False}
(BASE/'acquisition_vehicle_manifest.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
print(json.dumps(row),flush=True)
