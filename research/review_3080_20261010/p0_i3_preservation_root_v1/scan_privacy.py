"""P0-I3 preservation: privacy / sensitive-information scan (read-only).

Runs BEFORE any copy or push.  Prints only pattern names, counts and MASKED samples; it never
echoes a candidate secret.  The SFTP credential containment check prints booleans only.

Exit codes: 0 = clear to preserve (no blocking finding)
            3 = BLOCKING finding (credential/secret material) -> must not push
"""
import json
import re
import sys
from pathlib import Path

def _find_root(start):
    """Repo root = nearest ancestor (or self) containing MASKCO_code/."""
    for cand in (start, *start.parents):
        if (cand / 'MASKCO_code').is_dir():
            return cand
    raise RuntimeError('repo root with MASKCO_code/ not found above %s' % start)


ROOT = _find_root(Path(__file__).resolve().parent)

TARGETS = [
    'C-VRP_Cold-chainVehicleRoutingProblem/results/i3_registered_1008_v5/'
    'LOCAL_128_STAGE_ACCEPTANCE_20261009.json',
    'C-VRP_Cold-chainVehicleRoutingProblem/results/i3_registered_1008_v5/'
    'INDEPENDENT_128_ACTUAL_IDENTITY_CHECK_20261009.json',
    'C-VRP_Cold-chainVehicleRoutingProblem/results/i3_online_recovery_1009_v1/'
    'INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009.json',
]

# BLOCKING patterns: credentials or secret material must never reach a public repo.
BLOCKING = [
    ('private_key_block', re.compile(r'BEGIN [A-Z ]*PRIVATE KEY')),
    ('ssh_public_key', re.compile(r'ssh-(rsa|ed25519|dss)\s')),
    ('password_field', re.compile(r'(?i)"(password|passwd|pwd)"\s*:')),
    ('secret_or_token_field', re.compile(r'(?i)"[^"]*(secret|token|api[_-]?key|access[_-]?key)'
                                         r'[^"]*"\s*:')),
    ('auth_header', re.compile(r'(?i)\b(authorization|bearer)\b\s*[:=]')),
    ('conn_string', re.compile(r'(?i)(postgres|mysql|mongodb|redis|sftp|ssh)://')),
    ('bearer_like_token', re.compile(r'\b(gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})\b')),
]

# REPORT-ONLY patterns: already published elsewhere in this repository (documented host path),
# so they are reported but do not by themselves block preservation.
REPORT_ONLY = [
    ('email', re.compile(r'[\w.+-]+@[\w-]+\.[\w.]{2,}')),
    ('ipv4', re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')),
    ('posix_home_path', re.compile(r'/home/[A-Za-z0-9_.-]+')),
    ('windows_user_path', re.compile(r'[A-Za-z]:\\\\?Users\\\\?[^\\\\"\s]+')),
    ('hostname_like', re.compile(r'(?i)\b(user|localhost|node\d+|gpu\d+)\b')),
]


def mask(s):
    s = str(s)
    if len(s) <= 8:
        return s[:2] + '***'
    return s[:4] + '***' + s[-2:]


def sftp_containment(text):
    """Does the text contain the configured SFTP host/user/password? Booleans only."""
    cfg = ROOT / '.vscode' / 'sftp.json'
    out = {'config_present': cfg.is_file(), 'host_in_text': None, 'username_in_text': None,
           'password_in_text': None}
    if not cfg.is_file():
        return out
    try:
        d = json.loads(cfg.read_text(encoding='utf-8'))
    except Exception as exc:
        out['config_error'] = repr(exc)
        return out
    for key, field in (('host_in_text', 'host'), ('username_in_text', 'username'),
                       ('password_in_text', 'password')):
        val = d.get(field)
        out[key] = bool(val) and (str(val) in text) if val else None
    return out


def main():
    findings = {'stage': 'P0_I3_preservation_privacy_scan', 'targets': {}, 'blocking': [],
                'report_only': [], 'verdict': 'clear_to_preserve'}
    for rel in TARGETS:
        p = ROOT / rel
        if not p.is_file():
            findings['blocking'].append({'file': rel, 'reason': 'file missing'})
            findings['targets'][rel] = {'exists': False}
            continue
        text = p.read_text(encoding='utf-8', errors='replace')
        entry = {'exists': True, 'bytes': p.stat().st_size, 'chars': len(text),
                 'blocking_hits': {}, 'report_hits': {}, 'sftp_containment': sftp_containment(text)}
        for name, rx in BLOCKING:
            hits = rx.findall(text)
            if hits:
                entry['blocking_hits'][name] = {'count': len(hits),
                                                'sample_masked': mask(hits[0])}
                findings['blocking'].append({'file': rel, 'pattern': name,
                                             'count': len(hits)})
        for name, rx in REPORT_ONLY:
            hits = rx.findall(text)
            if hits:
                uniq = sorted({h if isinstance(h, str) else str(h) for h in hits})
                entry['report_hits'][name] = {'count': len(hits), 'unique': len(uniq),
                                              'samples_masked': [mask(u) for u in uniq[:3]]}
                findings['report_only'].append({'file': rel, 'pattern': name,
                                                'count': len(hits), 'unique': len(uniq)})
        findings['targets'][rel] = entry

    # SFTP credential containment is blocking if true
    for rel, e in findings['targets'].items():
        sc = e.get('sftp_containment') or {}
        if sc.get('password_in_text') is True:
            findings['blocking'].append({'file': rel, 'pattern': 'sftp_password_contained',
                                         'count': 1})
    if findings['blocking']:
        findings['verdict'] = 'BLOCKED_do_not_push'
    print(json.dumps(findings, ensure_ascii=False, indent=2)[:6000])
    out = Path(__file__).resolve().parent / 'P0_I3_PRIVACY_SCAN.json'
    out.write_text(json.dumps(findings, ensure_ascii=False, indent=2) + '\n',
                   encoding='utf-8', newline='\n')
    return 3 if findings['blocking'] else 0


if __name__ == '__main__':
    sys.exit(main())
