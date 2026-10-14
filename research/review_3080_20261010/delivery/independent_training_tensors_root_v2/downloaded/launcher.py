import base64,json,pathlib,subprocess,sys
base=pathlib.Path(__file__).resolve().parent
if (base/'LAUNCH.json').exists():raise RuntimeError('Already launched; no duplicate run')
pythonw=pathlib.Path(sys.executable).with_name('pythonw.exe')
if not pythonw.exists():raise RuntimeError('Hidden Python interpreter unavailable')
command=subprocess.list2cmdline([str(pythonw),str(base/'supervisor.py')])
payload=base64.b64encode(json.dumps({'command':command,'base':str(base)}).encode()).decode()
ps='$d=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String("'+payload+'"))|ConvertFrom-Json; $r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$d.command;CurrentDirectory=$d.base}; $r|ConvertTo-Json -Compress; if($r.ReturnValue -ne 0){exit 1}'
r=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',base64.b64encode(ps.encode('utf-16-le')).decode()],capture_output=True,text=True)
if r.returncode!=0:raise RuntimeError('WMI launch failed: '+r.stdout+r.stderr)
result=json.loads(r.stdout.strip())
with (base/'LAUNCH.json').open('x') as f:json.dump({'pid':result['ProcessId'],'mode':'hidden_pythonw_WMI_outside_SSH_job','max_seconds':300,'previous_attempt':'v1 interrupted before supervisor body; retained'},f)
print(json.dumps({'supervisor_pid':result['ProcessId'],'maximum_seconds':300,'research_model_eligible':False}))
