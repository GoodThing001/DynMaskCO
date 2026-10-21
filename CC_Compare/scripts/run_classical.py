"""Delegate to the existing strict-profit classical baseline entry point."""
from pathlib import Path
import runpy
import sys

workspace = next(p for p in Path(__file__).resolve().parents
                 if (p / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/project_paths.py').is_file())
entry = workspace / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py'
sys.argv = [str(entry), 'classical', *sys.argv[1:]]
runpy.run_path(str(entry), run_name='__main__')
