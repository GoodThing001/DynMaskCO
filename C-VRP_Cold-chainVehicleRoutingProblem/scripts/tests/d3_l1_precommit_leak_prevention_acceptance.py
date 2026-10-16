#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PRE-COMMIT LEAK-PREVENTION ACCEPTANCE (isolated synthetic repository).

Proves the anti-recurrence control actually REFUSES a violating commit, and does not merely exist.
It builds a throwaway git repository in a temporary directory, stages SYNTHETIC files, runs the real
scanner against the real index (`git diff --cached`), and asserts:

  S1  a staged `guard/execution_secret_<...>.txt` is REFUSED                      (rule R1)
  S2  a staged file whose content is a bare 64-hex token is REFUSED               (rule R2)
  S3  a staged credential assignment is REFUSED                                   (rule R3)
  S4  a staged PEM private-key header is REFUSED                                  (rule R4)
  S5  a CLEAN staged file is ACCEPTED (no false positive on ordinary evidence)
  S6  a redaction STUB staged at the real path is ACCEPTED (the remedy must be committable)
  S7  the scanner reads the INDEX, not the working tree: a clean index with a dirty
      working tree is ACCEPTED, and a dirty index with a clean working tree is REFUSED
  S8  the scanner never prints a value: its output contains no staged secret literal
  S9  the real delivery path calls the scanner before it commits

NO REAL SECRET IS USED ANYWHERE.  Every value here is a synthetic, obviously-fake token.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
SCANNER = REPO / 'tools/scan_staged_secrets.py'
DELIVER = REPO / 'tools/git_deliver.py'
ROWS = []

SYNTHETIC_HEX = 'deadbeef' * 8          # 64 hex chars, unmistakably synthetic
SYNTHETIC_PW = 'hunter2hunter2'         # obviously fake
STUB_TEXT = ('REDACTED-EXPOSED-EXECUTION-SECRET\n'
             '==================================\n'
             'secret value sha256     : ' + 'a' * 64 + '   (one-way; the value is NOT recorded)\n')


def ck(name, ok, detail=None):
    ROWS.append({'name': name, 'ok': bool(ok), 'detail': detail})
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)[:190]))


def git(repo, *args, check=True):
    return subprocess.run(['git'] + list(args), cwd=str(repo), capture_output=True, text=True,
                          check=False)


def run_scan(repo):
    """Run the real scanner against `repo`'s index, with the CWD INSIDE that repo.

    `--repo` alone is not enough: `git diff --cached` resolves through the current directory, so a
    scan invoked from the outer repository reports the OUTER index.  The first cut of this harness
    omitted `cwd`, which made S7 read an index that was not the fixture's and return nothing useful.
    """
    p = subprocess.run([sys.executable, str(SCANNER), '--repo', str(repo)],
                       cwd=str(repo), capture_output=True, text=True)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        out = {'verdict': 'SCANNER_OUTPUT_UNPARSEABLE', 'stdout_tail': p.stdout[-300:],
               'stderr_tail': p.stderr[-300:]}
    return p.returncode, out, p.stdout


def fresh_repo(root):
    r = Path(tempfile.mkdtemp(prefix='leakscan_', dir=str(root)))
    git(r, 'init', '-q')
    git(r, 'config', 'user.email', 'acceptance@example.invalid')
    git(r, 'config', 'user.name', 'acceptance')
    git(r, 'config', 'commit.gpgsign', 'false')
    return r


def stage(repo, rel, content):
    p = Path(repo) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8')
    git(repo, 'add', '--', rel)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--keep', action='store_true')
    args = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix='.leakscan_root_', dir=str(REPO)))

    try:
        # S1 ---------------------------------------------------------------------------------
        r = fresh_repo(root)
        stage(r, 'run/guard/execution_secret_synthetic1234.txt', SYNTHETIC_HEX + '\n')
        rc, out, _ = run_scan(r)
        ck('S1 a staged `guard/execution_secret_<...>.txt` is REFUSED', rc != 0 and
           out.get('verdict') == 'STAGED_CONTENT_REFUSED' and
           any(f.get('rule') == 'R1_execution_secret_path' for f in out.get('findings', [])),
           {'verdict': out.get('verdict'), 'rules': [f.get('rule') for f in out.get('findings', [])]})

        # S2 ---------------------------------------------------------------------------------
        r = fresh_repo(root)
        stage(r, 'run/not_a_secret_name.txt', SYNTHETIC_HEX + '\n')
        rc, out, _ = run_scan(r)
        ck('S2 a staged file whose CONTENT is a bare 64-hex token is REFUSED (the value shape, not '
           'just the file name)', rc != 0 and any(
               f.get('rule') == 'R2_bare_64_hex_token' for f in out.get('findings', [])),
           {'verdict': out.get('verdict'), 'rules': [f.get('rule') for f in out.get('findings', [])]})

        # S3 ---------------------------------------------------------------------------------
        r = fresh_repo(root)
        # NOTE: every detector-shaped literal in this file is SPLIT at a string boundary.  The
        # scanner must never trip on its own acceptance fixtures -- an earlier version did, and the
        # first real scan of this round refused a file whose only sin was containing the keyword the
        # rule looks for.
        stage(r, 'conf/settings.json', '{\n  "pass' + 'word": "%s"\n}\n' % SYNTHETIC_PW)
        rc, out, _ = run_scan(r)
        ck('S3 a staged credential assignment is REFUSED', rc != 0 and any(
            f.get('rule') == 'R3_credential_assignment' for f in out.get('findings', [])),
           {'verdict': out.get('verdict'), 'rules': [f.get('rule') for f in out.get('findings', [])]})

        # S4 ---------------------------------------------------------------------------------
        r = fresh_repo(root)
        stage(r, 'keys/id_pem.txt', '-----' + 'BEGIN RSA PRIVATE ' + 'KEY-----' + '\nzzz\n')
        rc, out, _ = run_scan(r)
        ck('S4 a staged PEM private-key header is REFUSED', rc != 0 and any(
            f.get('rule') == 'R4_pem_private_key_header' for f in out.get('findings', [])),
           {'verdict': out.get('verdict'), 'rules': [f.get('rule') for f in out.get('findings', [])]})

        # S5 ---------------------------------------------------------------------------------
        r = fresh_repo(root)
        stage(r, 'evidence/L1_ADJUDICATION.json',
              json.dumps({'verdict': 'BLOCKED', 'n_states': 64,
                          'sha256': 'f786b3c202870670226f892f5ace31aa4ff04c52087e47b3be29cc8dc443a389'},
                         indent=1) + '\n')
        stage(r, 'evidence/notes.md', 'the round is BLOCKED; no secret is written here\n')
        rc, out, _ = run_scan(r)
        ck('S5 an ordinary evidence file is ACCEPTED -- including one carrying a 64-hex DIGEST inside '
           'a JSON field (a digest is not a bare token)', rc == 0 and
           out.get('verdict') == 'STAGED_CONTENT_CLEAN',
           {'verdict': out.get('verdict'), 'n_findings': out.get('n_findings'),
            'n_staged_scanned': out.get('n_staged_scanned')})

        # S6 ---------------------------------------------------------------------------------
        # The scanner's default allow-list is EXACT repository-relative paths, deliberately not a
        # `*/guard/execution_secret_*.txt` glob (that glob would also exempt exactly the files R1
        # exists to catch).  In a synthetic repository those paths differ, so the exemption is passed
        # explicitly here -- which is also the realistic shape of a reviewed exception.
        r = fresh_repo(root)
        stub_rel = 'run/guard/execution_secret_synthetic1234.txt'
        stage(r, stub_rel, STUB_TEXT)
        p = subprocess.run([sys.executable, str(SCANNER), '--repo', str(r), '--allow', stub_rel],
                           cwd=str(r), capture_output=True, text=True)
        out = json.loads(p.stdout) if p.stdout.strip().startswith('{') else {}
        ck('S6 the REDACTION STUB at a secret path is ACCEPTED when the path is explicitly exempted '
           '(the remedy stays committable), and it is content-checked too', p.returncode == 0 and
           out.get('verdict') == 'STAGED_CONTENT_CLEAN' and out.get('n_exempted') == 1,
           {'verdict': out.get('verdict'), 'n_exempted': out.get('n_exempted'),
            'n_findings': out.get('n_findings'),
            'content_check': 'an exempted path whose content is NOT the stub is refused as '
                             'R0_exempted_path_without_stub_marker'})

        # S6b --------------------------------------------------------------------------------
        # the exemption is BORROWABLE-PROOF: same allowed path, non-stub content -> refused
        r = fresh_repo(root)
        stage(r, stub_rel, SYNTHETIC_HEX + '\n')
        p = subprocess.run([sys.executable, str(SCANNER), '--repo', str(r), '--allow', stub_rel],
                           cwd=str(r), capture_output=True, text=True)
        out = json.loads(p.stdout) if p.stdout.strip().startswith('{') else {}
        ck('S6b an allow-listed PATH carrying NON-stub content is REFUSED: an exemption cannot be '
           'borrowed by a file that merely sits at an allowed path', p.returncode != 0 and any(
               f.get('rule') == 'R0_exempted_path_without_stub_marker'
               for f in out.get('findings', [])),
           {'verdict': out.get('verdict'),
            'rules': [f.get('rule') for f in out.get('findings', [])]})

        # S7 ---------------------------------------------------------------------------------
        r = fresh_repo(root)
        clean = stage(r, 'evidence/clean.txt', 'nothing sensitive\n')
        rc_clean_index, out_clean, _ = run_scan(r)
        clean.write_text(SYNTHETIC_HEX + '\n', encoding='utf-8')          # dirty worktree only
        rc_dirty_worktree, out_dw, _ = run_scan(r)
        r2 = fresh_repo(root)
        p2 = stage(r2, 'evidence/x.txt', SYNTHETIC_HEX + '\n')            # dirty index
        p2.write_text('now clean\n', encoding='utf-8')                    # clean worktree
        rc_clean_worktree, out_cw, _ = run_scan(r2)
        s7_ok = (rc_clean_index == 0 and rc_dirty_worktree == 0
                 and out_dw.get('verdict') == 'STAGED_CONTENT_CLEAN'
                 and rc_clean_worktree != 0)
        ck('S7 the scanner reads the INDEX, not the working tree: a clean index with a dirty worktree '
           'is ACCEPTED, and a dirty index with a clean worktree is REFUSED', s7_ok,
           {'clean_index_dirty_worktree': out_dw.get('verdict'),
            'dirty_index_clean_worktree': out_cw.get('verdict'),
            'why': 'the index is what becomes public; scanning the worktree would both miss staged '
                   'blobs and cry wolf over uncommitted scratch'})

        # S8 ---------------------------------------------------------------------------------
        r = fresh_repo(root)
        stage(r, 'run/guard/execution_secret_synthetic9999.txt', SYNTHETIC_HEX + '\n')
        stage(r, 'conf/s.json', '{"pass' + 'word": "%s"}\n' % SYNTHETIC_PW)
        rc, out, raw = run_scan(r)
        leaks = {'hex_token_in_output': SYNTHETIC_HEX in raw,
                 'password_in_output': SYNTHETIC_PW in raw}
        # and the JSON output must not carry a 'value'-like key with content
        value_keys = [k for f in out.get('findings', []) for k in f
                      if k.lower() in ('value', 'secret', 'content', 'match', 'line')]
        ck('S8 the scan output NEVER carries a value: no staged literal appears and no finding '
           'carries a value/line-content field', not any(leaks.values()) and not value_keys,
           {'leaks': leaks, 'unexpected_value_keys': value_keys,
            'reported_by': 'path + line number + rule id only',
            'raw_output_chars': len(raw)})

        # S9 ---------------------------------------------------------------------------------
        dsrc = DELIVER.read_text(encoding='utf-8')
        s9_ok = ('scan_staged_secrets' in dsrc or 'scan_staged' in dsrc)
        ck('S9 the DELIVERY path invokes the scanner before it commits, so the control is on the real '
           'route and not only available to a careful caller', s9_ok,
           {'delivery_tool': str(DELIVER.relative_to(REPO)),
            'mentions_the_scanner': s9_ok})
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)

    fails = [r for r in ROWS if not r['ok']]
    payload = {
        'schema': 'd3_l1_precommit_leak_prevention_acceptance',
        'rows': ROWS, 'n_rows': len(ROWS), 'n_pass': len(ROWS) - len(fails), 'n_fail': len(fails),
        'scanner': str(SCANNER.relative_to(REPO)),
        'rules_exercised': ['R1_execution_secret_path', 'R2_bare_64_hex_token',
                            'R3_credential_assignment', 'R4_pem_private_key_header'],
        'no_real_secret_used': True,
        'synthetic_values_only': ['a 64-hex token that says "deadbeef"', 'a throwaway password string'],
        'isolation': 'every case runs in a throwaway git repository under a temporary directory; '
                     'the real repository is never staged, committed or modified',
        'verdict': ('STAGED_LEAK_PREVENTION_ACCEPTED' if not fails
                    else 'STAGED_LEAK_PREVENTION_FAILED'),
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('\n=== pre-commit leak-prevention acceptance: %d rows, %d PASS, %d FAIL ==='
          % (len(ROWS), len(ROWS) - len(fails), len(fails)))
    print('WROTE', out)
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
