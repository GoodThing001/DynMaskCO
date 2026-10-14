"""Read-only identity check for this round's own process trees."""
import json
import psutil

rows=[];unreadable=0
for proc in psutil.process_iter(['pid','name','cmdline','create_time']):
    try:
        info=proc.info
        args=info.get('cmdline') or []
        if any('qualify_pickup_strong_owner_v1.py' in a or
               'qualify_pickup_strong_owner_v2.py' in a for a in args):
            rows.append(dict(pid=proc.pid,created=info['create_time'],cmdline=args))
    except (psutil.NoSuchProcess,psutil.AccessDenied):
        unreadable+=1
print(json.dumps(dict(matching=rows,unreadable_processes=unreadable)))
