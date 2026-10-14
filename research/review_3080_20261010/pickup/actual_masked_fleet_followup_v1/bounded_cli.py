"""Hard900s overall bound with only own descendant processes killed/reaped."""
import argparse,subprocess,sys
from pathlib import Path
import psutil
def run(argv):
    p=subprocess.Popen([sys.executable,'-B',str(Path(__file__).with_name('run_followup.py'))]+argv)
    try:return p.wait(timeout=900)
    except subprocess.TimeoutExpired:
        children=psutil.Process(p.pid).children(recursive=True)
        for child in children:
            try:child.kill()
            except psutil.NoSuchProcess:pass
        p.kill();p.wait(timeout=10);psutil.wait_procs(children,timeout=10)
        print('Bounded engineering runner exceeded900s; own children killed/reaped',file=sys.stderr)
        return 124
if __name__=='__main__':sys.exit(run(sys.argv[1:]))
