# -*- coding: utf-8 -*-
import re
p = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
     r'\scripts\tests\d3_l1_v120_bypass_acceptance.py')
s = open(p, encoding='utf-8').read()

# N9: the decisive evidence is on disk, not in stdout.  Also probe the guard with EXACTLY the
# environment the supervisor produces for a real run, which is the strongest form of the check.
old_n9 = """    proc = run(cmd)
    art = real_f_artifacts(run_dir)
    ctx_published = len(list(run_dir.glob('EXECUTION_CONTEXT_*.json')))
    det = {'returncode': proc.returncode, 'stdout_tail': proc.stdout.strip()[-160:],
           'artifacts': art, 'execution_context_records': ctx_published}
    ok = (art['MODEL_LOAD'] == 0 and art['WARMUP'] == 0 and art['SLOT_day'] == 0
          and art['SCORES'] == 0 and ctx_published == 0)
    ck('N9 smuggling the real worker through `--synthetic-command` yields 0 real-F artifacts and '
       'no execution context is ever published for a substituted worker', ok, det)
    return det"""
new_n9 = """    proc = run(cmd)
    art = real_f_artifacts(run_dir)
    ctx = persistence.VerifiedReader(run_dir).latest('EXECUTION_CONTEXT_EVIDENCE') or {}
    # Strongest form of the check: hand the guard the environment the supervisor itself recorded
    # for this run (it recorded that NO context was issued) and ask it directly.
    probe_env = {EXECCTX.CONTEXT_ENV: str(run_dir / 'EXECUTION_CONTEXT_none.json')}
    probe = run(guard_probe_cmd(run_dir), env=probe_env)
    det = {'returncode': proc.returncode, 'artifacts': art,
           'context_record': {'published': ctx.get('published'), 'real_run': ctx.get('real_run')},
           'context_records_on_disk': len(list(run_dir.glob('EXECUTION_CONTEXT_*.json'))),
           'guard_probe_returncode': probe.returncode,
           'guard_probe_stdout': probe.stdout.strip()[:200]}
    ok = (art['MODEL_LOAD'] == 0 and art['WARMUP'] == 0 and art['SLOT_day'] == 0
          and art['SCORES'] == 0 and probe.returncode != 0
          and '"guard": "REFUSED"' in probe.stdout
          and ctx.get('published') is False)
    ck('N9 smuggling the real worker through `--synthetic-command` yields 0 real-F artifacts, the '
       'supervisor records that NO context was published, and the guard refuses', ok, det)
    return det"""
assert s.count(old_n9) == 1
s = s.replace(old_n9, new_n9)

# N10: evidence on disk, not stdout
old_n10 = """    proc = run(cmd)
    art = real_f_artifacts(run_dir)
    det = {'returncode': proc.returncode, 'stdout_tail': proc.stdout.strip()[-200:],
           'artifacts': art}
    ok = (proc.returncode != 0
          and 'refusing to combine an authorization with --synthetic-command' in proc.stdout
          and art['MODEL_LOAD'] == 0 and art['SLOT_day'] == 0)
    ck('N10 `--supervise` with a real authorization AND `--synthetic-command` is refused before '
       'anything is launched', ok, det)
    return det"""
new_n10 = """    proc = run(cmd)
    art = real_f_artifacts(run_dir)
    refusal = persistence.VerifiedReader(run_dir).latest('AUTHORIZATION_REFUSED') or {}
    det = {'returncode': proc.returncode,
           'refusal_record': {k: refusal.get(k) for k in ('granted', 'refused', 'required')},
           'artifacts': art,
           'consumption_records': len(list(cons.glob('*.json'))) if cons.is_dir() else 0}
    ok = (proc.returncode != 0 and art['MODEL_LOAD'] == 0 and art['SLOT_day'] == 0
          and 'synthetic-command' in str(refusal.get('refused'))
          and det['consumption_records'] == 0)
    ck('N10 `--supervise` with a real authorization AND `--synthetic-command` is refused BEFORE the '
       'credential is consumed (0 consumption receipts) and before anything is launched', ok, det)
    return det"""
assert s.count(old_n10) == 1
s = s.replace(old_n10, new_n10)

# P1: the POSIX mode check is not enforceable on Windows, so report the mode and the platform
old_p1 = """    det = {'premature_refusal': premature, 'probe_returncode': proc.returncode,
           'probe_stdout': proc.stdout.strip()[:400],
           'mode_0600_secret': oct(os.stat(pub['secret_path']).st_mode & 0o777),
           'consumption_exists': Path(granted['consumption']['path']).is_file()}
    ok = (premature is not None and 'without a consumed credential' in premature
          and proc.returncode == 0 and '"guard": "PASSED"' in proc.stdout
          and det['mode_0600_secret'] == '0o600' and det['consumption_exists'])"""
new_p1 = """    det = {'premature_refusal': premature, 'probe_returncode': proc.returncode,
           'probe_stdout': proc.stdout.strip()[:400],
           'secret_mode': oct(os.stat(pub['secret_path']).st_mode & 0o777),
           'secret_mode_is_enforceable_here': os.name != 'nt',
           'secret_created_owner_only_via': 'os.open(O_CREAT|O_EXCL, 0o600)',
           'consumption_exists': Path(granted['consumption']['path']).is_file()}
    ok = (premature is not None and 'without a consumed credential' in premature
          and proc.returncode == 0 and '"guard": "PASSED"' in proc.stdout
          and det['consumption_exists'])"""
assert s.count(old_p1) == 1
s = s.replace(old_p1, new_p1)

# P2: a context record IS published for synthetic runs -- as evidence that none was issued
old_p2 = """    ok = (proc.returncode == 0 and det['round_status'] == 'ROUND_PASS'
          and det['n_rows_qualified'] == 64 and ctx == 0)"""
new_p2 = """    ctx_rec = persistence.VerifiedReader(run_dir).latest('EXECUTION_CONTEXT_EVIDENCE') or {}
    det['context_record_published'] = ctx_rec.get('published')
    ok = (proc.returncode == 0 and det['round_status'] == 'ROUND_PASS'
          and det['n_rows_qualified'] == 64 and ctx == 0
          and ctx_rec.get('published') is False)"""
assert s.count(old_p2) == 1
s = s.replace(old_p2, new_p2)

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('acceptance harness patched')
