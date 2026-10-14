"""Launch the registered hidden supervisor once; no duplicate or retry path."""
import base64,json,subprocess,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
from action_head_runtime_v1 import sha,save

def main():
    reg=json.loads((BASE/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    if not reg['execution_authorized'] or sha(__file__)!=reg['package_files'][Path(__file__).name]:raise ValueError('launch registration')
    for n,h in reg['package_files'].items():
        if sha(BASE/n)!=h:raise ValueError('package:'+n)
    save(BASE,'LAUNCH_ATTEMPT.json',dict(created_epoch=time.time(),registration_sha256=sha(BASE/'EXECUTION_REGISTRATION.json')))
    pythonw=Path(sys.executable).with_name('pythonw.exe')
    if not pythonw.is_file():raise RuntimeError('hidden interpreter missing')
    command=subprocess.list2cmdline([str(pythonw),'-B',str(BASE/'supervise_delivery_actual_score_chain_v1.py')])
    payload=base64.b64encode(json.dumps(dict(command=command,base=str(BASE))).encode()).decode()
    ps='$d=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("'+payload+'"))|ConvertFrom-Json; $r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$d.command;CurrentDirectory=$d.base}; $r|ConvertTo-Json -Compress; if($r.ReturnValue -ne 0){exit 1}'
    r=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(ps.encode('utf-16-le')).decode()],capture_output=True,text=True,timeout=30)
    save(BASE,'LAUNCH_TRANSPORT.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode!=0:raise RuntimeError('uncertain launch; inspect only, no retry')
    observed=json.loads(r.stdout.strip())
    save(BASE,'LAUNCH.json',dict(pid=observed['ProcessId'],hidden_pythonw=True,registration_sha256=sha(BASE/'EXECUTION_REGISTRATION.json')))
    print(json.dumps(dict(pid=observed['ProcessId'],cpu=15,scientific_family3_started=False)))

if __name__=='__main__':main()
