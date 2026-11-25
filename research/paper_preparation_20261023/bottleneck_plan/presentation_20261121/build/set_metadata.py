from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

base = Path(__file__).resolve().parent
source = base / 'candidate.pptx'
target = base / 'candidate_metadata.pptx'
with ZipFile(source) as src, ZipFile(target, 'w') as dst:
    for member in src.infolist():
        content = src.read(member.filename)
        if member.filename == 'docProps/core.xml':
            root = ET.fromstring(content)
            for element in list(root):
                name = element.tag.rsplit('}', 1)[-1]
                if name in ('creator', 'lastModifiedBy'):
                    root.remove(element)
                elif name == 'title':
                    element.text = '基于MASKCO掩码重构与经济分车的静态冷链多车配送优化'
            content = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        dst.writestr(member, content)
target.replace(source)
print('Presentation metadata saved.')
