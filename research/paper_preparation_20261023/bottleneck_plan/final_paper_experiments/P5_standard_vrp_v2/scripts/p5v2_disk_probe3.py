import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport
transport.record = lambda *a, **k: None
client, _ = transport.connect(ROOT / ".vscode/sftp.json")
try:
    cmd = "; ".join([
        "echo '[mounts]'; df -h | grep -vE 'tmpfs|udev|loop'",
        "echo '[tmp inodes]'; df -i /home | tail -1",
        "echo '[writable candidates]'; for d in /tmp /var/tmp /data /scratch /mnt /home/hzeng/project; do [ -w $d ] && echo \"$d ok\"; done",
    ])
    print(transport.remote_exec(client, cmd))
finally:
    client.close()
