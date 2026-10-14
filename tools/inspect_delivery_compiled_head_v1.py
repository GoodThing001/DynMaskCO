"""Read exact new model-stage receipts without printing its large evidence rows."""
import datetime
import json
import transport_delivery_compiled_head_v1 as transport


def main():
    transport.verify_local()
    r=transport.call('compiled_head_remote_status_v1.py')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    transport.save('COMPACT_INSPECTION_'+stamp+'.json',r)
    if r['exit_code']!=0:raise RuntimeError('Readonly stage status failed')
    x=json.loads(r['stdout']);receipts=x['receipts']
    audit=receipts.get('INDEPENDENT_TIMED_REQUEST_AUDIT.json',{})
    print(json.dumps(dict(complete=x['complete'],
        owned_processes=[dict(pid=p['pid'],ppid=p['ppid'],rss=p['rss'],affinity=p['affinity'],script=p['cmdline'][-1]) for p in x['owned_processes']],
        run_exit=receipts.get('RUN_EXIT.json'),audit_exit=receipts.get('AUDIT_RUN_EXIT.json'),chain_exit=receipts.get('CHAIN_EXIT.json'),
        failure=receipts.get('FAILURE.json'),audit_failure=receipts.get('TIMED_AUDIT_FAILURE.json'),
        audit_checks=audit.get('checks_count'),audit_failures=audit.get('failed'),audit_failure_names=audit.get('failure_names'),
        file_count=len(x['files']))))


if __name__=='__main__':main()
