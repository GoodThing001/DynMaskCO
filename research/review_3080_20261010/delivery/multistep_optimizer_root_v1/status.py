import json,pathlib,psutil
base=pathlib.Path(__file__).resolve().parent
r={'base':str(base),'finished':(base/'RUN_EXIT.json').exists()}
if r['finished']:r['run_exit']=json.loads((base/'RUN_EXIT.json').read_text())
else:
 try:
  pid=json.loads((base/'LAUNCH.json').read_text())['pid'];p=psutil.Process(pid)
  r['owned_process_tree']=[{'pid':q.pid,'name':q.name(),'cmdline':q.cmdline(),'cpu_affinity':q.cpu_affinity(),'rss':q.memory_info().rss,'status':q.status()} for q in [p]+p.children(recursive=True)]
 except (OSError,psutil.Error) as e:r['process_error']=type(e).__name__
print(json.dumps(r))
