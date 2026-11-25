import os
import runpy
import shutil
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
skill = Path(r'C:\Users\mmc\.codex\plugins\cache\openai-primary-runtime\presentations\26.1119.11507\skills\presentations\container_tools')

def publish_copy(source, destination, **kwargs):
    with open(source, 'rb') as src, open(destination, 'xb') as dst:
        shutil.copyfileobj(src, dst)

os.link = publish_copy
sys.path.insert(0, str(skill))
sys.argv = [str(skill / 'materialize_literal_chart_workbooks.py'),
            str(base / 'candidate.pptx'), str(base / 'candidate_complete.pptx'),
            '--workspace', str(base.parent), '--receipt', str(base / 'chart_data_ready.json')]
runpy.run_path(sys.argv[0], run_name='__main__')
