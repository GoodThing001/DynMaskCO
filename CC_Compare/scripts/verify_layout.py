"""Check the current ten-method directory without running experiments."""
from pathlib import Path
import runpy
import sys

workspace = next(p for p in Path(__file__).resolve().parents
                 if (p / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/project_paths.py').is_file())
entry = workspace / 'C-VRP_Cold-chainVehicleRoutingProblem/tools/maintenance/verify_comparison_sources.py'
sys.path.insert(0, str(entry.parent))
sys.argv = [str(entry), *sys.argv[1:]]
runpy.run_path(str(entry), run_name='__main__')
