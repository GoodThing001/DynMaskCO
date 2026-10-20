"""Read-only inventory of the configured user-owned Windows project."""
from pathlib import Path
import base64
import json
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import paramiko

ROOT=Path(__file__).resolve().parents[4]
settings=json.loads((ROOT/'.vscode/sftp.3080.json').read_text(encoding='utf-8-sig'))
client=paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect(settings['host'],port=settings.get('port',22),username=settings['username'],password=settings.get('password'),timeout=15,look_for_keys=False,allow_agent=False)
    sftp=client.open_sftp()
    remote=settings['remotePath'].rstrip('/')
    entries=sftp.listdir_attr(remote)
    top=[e.filename for e in entries if not any(t in e.filename.lower() for t in ('private','credential','secret','.git','.vscode'))]
    result={'configured_project':remote,'top_level':top}
    native=remote.lstrip('/').replace('/','\\')
    code=r'''$ErrorActionPreference='Stop'
$root='ROOT_VALUE'
$skip=@('.git','.hg','.svn','.vscode','.cursor','.claude','__pycache__','node_modules','.venv','venv','env','cache','.cache','wandb','checkpoints','private')
$queue=[System.Collections.Generic.Queue[object]]::new()
$queue.Enqueue(@{path=$root;depth=0})
$matches=[System.Collections.Generic.List[object]]::new()
$errors=[System.Collections.Generic.List[object]]::new()
$prunedRuns=[System.Collections.Generic.List[string]]::new()
$visited=0;$directories=0;$prunedDepth=0;$stopped=$false
$timer=[System.Diagnostics.Stopwatch]::StartNew()
while($queue.Count -gt 0){
 $current=$queue.Dequeue();$directories++
 try{$entries=Get-ChildItem -LiteralPath $current.path -Force -ErrorAction Stop}catch{$errors.Add(@{path=$current.path;type=$_.Exception.GetType().Name});continue}
 foreach($entry in $entries){
  $visited++;$name=$entry.Name;$lower=$name.ToLowerInvariant();$path=$entry.FullName
  if(($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){continue}
  if($skip -contains $lower -or $lower -match 'private|credential|secret'){continue}
  if($entry.PSIsContainer -and $name -match '^run0[1-5]$' -and $path -match 'static_rrp_20261020'){$prunedRuns.Add($path);continue}
  $direct=$name -match 'rrp|refrigerated|energies.?13.?6214|sol-D1-n9-np32'
  $inRrp=$current.path.Substring($root.Length) -match 'rrp|refrigerated'
  if($direct -or ($inRrp -and $name -match '\.(json|txt|cpp|h|mzn|py|md|zip)$')){$matches.Add(@{path=$path;directory=[bool]$entry.PSIsContainer;size=if($entry.PSIsContainer){$null}else{$entry.Length}})}
  if($entry.PSIsContainer){if($current.depth -ge 12){$prunedDepth++}else{$queue.Enqueue(@{path=$path;depth=($current.depth+1)})}}
 }
 if($visited -ge 180000 -or $timer.Elapsed.TotalSeconds -gt 22){$stopped=$true;break}
}
@{root=$root;max_depth=12;max_entries=180000;max_seconds=22;excluded=$skip;excluded_recent_runs=@('run01','run02','run03','run04','run05');visited_entries=$visited;visited_directories=$directories;pruned_depth=$prunedDepth;pruned_recent_runs=$prunedRuns;errors=$errors;matches=$matches;stopped_early=$stopped;elapsed_seconds=$timer.Elapsed.TotalSeconds}|ConvertTo-Json -Depth 8 -Compress
'''.replace('ROOT_VALUE',native.replace("'","''"))
    command='powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand '+base64.b64encode(code.encode('utf-16le')).decode('ascii')
    _,stdout,stderr=client.exec_command(command,timeout=35)
    out=stdout.read().decode('utf-8','replace').lstrip('\ufeff')
    err=stderr.read().decode('utf-8','replace')
    status=stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError('Remote read-only discovery failed: '+err[-2000:])
    result.update(json.loads(out))
    target=Path(__file__).parent/'windows_discovery.json'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'evidence':str(target),'visited_entries':result['visited_entries'],'stopped_early':result['stopped_early'],'matches':result['matches'],'errors':result['errors']},ensure_ascii=False))
finally:
    client.close()
