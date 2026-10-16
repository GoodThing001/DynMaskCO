#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""STAGED-CONTENT SECRET SCAN (anti-recurrence).

WHY THIS EXISTS
  Two 256-bit execution secrets were committed and pushed to the public repository inside raw run
  directories: `guard/execution_secret_<...>.txt`.  The delivery path used `git add -A` and had NO
  secret-shape check at all, so the leak was found by the project owner's review rather than by any
  check of mine.  This is that missing check.

WHAT IT DOES
  Scans the STAGED content of every file git is about to commit (`git diff --cached`), never the
  working tree, because the working tree is not what becomes public.  It refuses on:

    R1  any path matching `execution_secret_*` (the secret's own file shape)
    R2  any staged line that is exactly a bare 64-hex token (the secret's VALUE shape,
        which is how `secrets.token_hex(32)` writes it)
    R3  a high-signal credential assignment (`password=`, `secret:`, `api_key=`, ...) with a
        non-trivial literal value
    R4  a PEM private-key header
    R5  a staged entry whose blob could NOT be read, so the file was never actually scanned

  It is deliberately a REFUSAL, not a warning: a scan that only warns is the scan that was absent.

COVERAGE IS PART OF THE VERDICT (2026-10-16)
  R5 exists because this scanner had exactly that hole and reported `CLEAN` through it.  The staged
  name list was read without `-z`, so git C-quoted every non-ASCII path
  (`"docs/\345\275\223..."`); feeding that quoted string back to `git show :<name>` fails with
  return code 128, and the old code did `if raw is None: continue` -- a SILENT skip that was not
  even counted.  The delivery path therefore printed `secret scan: CLEAN (0 staged file(s) scanned)`
  on commits it had not looked at, and 2185 of this repository's tracked files live under non-ASCII
  directories (the whole `docs/<chinese>/...` research record).  Three things now make that
  impossible:

    * the staged name list is read with `-z`, so git emits raw, UNQUOTED bytes; and
    * an unreadable staged blob is a FINDING (R5), not a skip; and
    * the result reports `n_staged_entries` next to `n_staged_scanned`, so "0 scanned" can never be
      printed as if it were a clean bill of health.

  A control that silently narrows its own scope is worse than no control, because it is trusted.

WHAT IT DOES NOT DO
  It never prints, echoes, logs or stores a secret value -- a hit is reported by PATH and LINE NUMBER
  and by rule id only.  It makes no network call and writes nothing.

DELIBERATE EXEMPTIONS
  `--allow <path-glob>` exempts a path from R1..R4, for evidence that must stay public and carries no
  live credential.  The redaction STUBS are explicitly allowed by default: they are the remedy for the
  original leak, they contain only one-way digests, and refusing them would make the fix uncommittable.
  `--self-test` proves the rules fire, using SYNTHETIC throwaway values only.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve()
for _ in range(6):
    if (REPO_DEFAULT.parent / '.git').is_dir():
        REPO_DEFAULT = REPO_DEFAULT.parent
        break
    REPO_DEFAULT = REPO_DEFAULT.parent

BARE_HEX_64 = re.compile(r'^[0-9a-fA-F]{64}$')
BARE_HEX_64_IN_QUOTES = re.compile(r'^[\'"][0-9a-fA-F]{64}[\'"],?$')
# The detector's own patterns are SPLIT so this file cannot match itself.  A scanner that trips on
# its own source is a scanner whose first real run is a false positive -- exactly what happened when
# this file and the acceptance harness were first staged.  The same trick is used in the tests.
PEM_HEADER = re.compile('-----' + 'BEGIN [A-Z ]*PRIVATE ' + 'KEY-----')
ASSIGN = re.compile(
    # `["\']?` lets a JSON/YAML key be QUOTED.  With a bare leading `\b` the pattern never matched
    # `"password": "..."`, because there is no word boundary between a quote and `p` -- which meant
    # the single most likely real-world shape (a credential in a JSON config) went undetected.
    r'(?i)["\']?\b(pass(?:word|wd)?|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|'
    r'client[_-]?secret|auth(?:orization)?[_-]?token)\b["\']?\s*[:=]\s*'
    r'["\']?([^\s"\',;]{8,})["\']?')
# Values that are obviously NOT credentials: placeholders, templates, references, booleans.
# Every alternative is FULLY ANCHORED and self-contained.  An earlier version left a bare
# `[A-Za-z_][A-Za-z0-9_]*` alternative unanchored, which -- with `re.match` -- made the pattern
# accept ANY lowercase word that contained an underscore, silently exempting real-looking values.
BENIGN_VALUE = re.compile(
    r'(?i)^(?:'
    r'none|null|nil|true|false|yes|no'
    r'|redacted|placeholder|changeme|changeme_me|dummy|sample|example|test[_-]?only|not[_-]?set|'
    r'unspecified|omitted|withheld'
    r'|\{\{.*\}\}'                      # {{ vault_secret }} style template
    r'|\$\{.*\}'                        # ${VAR} interpolation
    r'|<[^>]*>'                         # <YOUR_KEY_HERE>
    r'|\*+|-+'
    r'|\.\.\.'
    r'|os\.environ.*|getenv.*|env\..*|process\.env.*|secrets\.[A-Za-z_]+.*'
    r'|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*'   # a bare attribute reference like cfg.token
    r')$')

# The ONLY paths allowed to carry the secret-file NAME.  Kept as EXACT repository-relative paths
# rather than a `*/guard/execution_secret_*.txt` glob: that glob also exempted any NEW file with the
# same shape, which is precisely what rule R1 exists to catch.  Both entries below are the redaction
# stubs -- the REMEDY for the original leak -- so refusing them would make the fix uncommittable.
DEFAULT_ALLOW = (
    'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1/'
    'l1_v2_execution_run/guard/execution_secret_12adf7457768bfb5.txt',
    'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1/'
    'v122_evidence/run_v122/guard/execution_secret_b3e36be45ceb6bb4.txt',
    'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1/'
    'l1_gates/L1_EXPOSED_SECRET_REDACTION.json',
)
# A staged secret-named file is accepted ONLY if its content is the redaction stub.  This makes the
# exemption content-based as well as path-based, so an exemption cannot be borrowed by a file that
# merely sits at an allowed path.
STUB_MARKER = 'REDACTED-EXPOSED-EXECUTION-SECRET'


def _git_bytes(repo, args):
    """Run git and return the RAW result.

    Bytes, never text: `text=True` decodes with the locale codec, which on Windows is a code page
    that mangles exactly the non-ASCII paths this module must handle.
    """
    return subprocess.run(['git'] + list(args), cwd=str(repo), capture_output=True)


def staged_files(repo):
    """Staged paths, raw and UNQUOTED.

    `-z` is load-bearing, not a style choice.  Without it git C-quotes any path that is not plain
    ASCII, and that quoted string is not a name this process can hand back to `git show :<name>` --
    which is how every non-ASCII staged file came to be silently skipped (see R5 above).

    `--diff-filter=ACMR` keeps the scope at "what this commit will ADD or CHANGE": a staged DELETION
    has no blob to scan, and scanning the whole index instead would mean re-reading every tracked
    file on every commit.
    """
    out = _git_bytes(repo, ['diff', '--cached', '--name-only', '-z', '--diff-filter=ACMR']).stdout
    return [p.decode('utf-8', 'surrogateescape') for p in out.split(b'\0') if p]


def staged_content(repo, path):
    """The staged blob's bytes -- NOT the working tree's."""
    out = _git_bytes(repo, ['show', ':%s' % path])
    if out.returncode != 0:
        return None
    return out.stdout


def staged_blob_mode(repo, path):
    """The index mode of a staged path, or None when the index has no entry for it.

    Used only to tell a gitlink (mode 160000, a submodule entry) from a blob: a gitlink legitimately
    has no blob of its own, so it must not be reported as an unreadable file.
    """
    out = _git_bytes(repo, ['ls-files', '-s', '-z', '--', path])
    if out.returncode != 0 or not out.stdout:
        return None
    head = out.stdout.split(b'\0', 1)[0]
    fields = head.split(b' ')
    if len(fields) < 2:
        return None
    return fields[0].decode('ascii', 'replace')


def scan_bytes(raw: bytes):
    """`[(rule, line_no)]` -- rule id and line number only; never the value."""
    hits = []
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('latin-1')
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if BARE_HEX_64.match(s) or BARE_HEX_64_IN_QUOTES.match(s):
            hits.append(('R2_bare_64_hex_token', i))
            continue
        if PEM_HEADER.search(line):
            hits.append(('R4_pem_private_key_header', i))
            continue
        m = ASSIGN.search(line)
        if m:
            # strip surrounding quotes BEFORE the benign test: leaving the trailing quote on makes
            # every quoted value look non-benign, and removing it is what lets `"{{ vault }}"` and
            # `"REDACTED"` be recognised as placeholders
            val = m.group(2).strip().strip('"').strip("'")
            if not BENIGN_VALUE.match(val):
                hits.append(('R3_credential_assignment', i))
    return hits


def scan(repo, allow):
    findings, checked, exempted = [], 0, []
    unreadable, gitlinks = [], []
    paths = staged_files(repo)
    for path in paths:
        raw = staged_content(repo, path)
        allowed_path = any(fnmatch(path, g) or fnmatch('/' + path, g) for g in allow)
        if allowed_path:
            # path exemption is NOT sufficient on its own: the content must be the redaction stub
            text = (raw or b'').decode('utf-8', 'replace')
            if STUB_MARKER in text:
                exempted.append({'path': path, 'reason': 'redaction stub (one-way digests only)'})
                continue
            findings.append({'path': path, 'rule': 'R0_exempted_path_without_stub_marker',
                             'lines': [], 'all_rules': ['R0_exempted_path_without_stub_marker'],
                             'note': 'an allow-listed path carried content that is NOT the redaction '
                                     'stub; the exemption is borrowed and is refused'})
            continue
        if fnmatch(Path(path).name, 'execution_secret_*'):
            findings.append({'path': path, 'rule': 'R1_execution_secret_path',
                             'lines': [], 'all_rules': ['R1_execution_secret_path'],
                             'note': 'a run-time secret file shape; never commit it'})
            continue
        if raw is None:
            if staged_blob_mode(repo, path) == '160000':
                gitlinks.append(path)          # a submodule entry has no blob of its own
                continue
            # NEVER `continue` silently here.  A staged entry this scan cannot read is an entry it
            # did not scan, and an uncounted skip is precisely how the non-ASCII path hole survived.
            unreadable.append(path)
            findings.append({'path': path, 'rule': 'R5_staged_blob_unreadable', 'lines': [],
                             'all_rules': ['R5_staged_blob_unreadable'],
                             'note': 'the staged blob could not be read, so this file was NOT '
                                     'scanned; refused rather than skipped'})
            continue
        checked += 1
        hits = scan_bytes(raw)
        if hits:
            findings.append({'path': path, 'rule': hits[0][0],
                             'lines': [h[1] for h in hits],
                             'all_rules': sorted({h[0] for h in hits}),
                             'note': 'value withheld by design'})
    return {'n_staged_entries': len(paths), 'n_staged_scanned': checked,
            'n_unreadable': len(unreadable), 'unreadable': unreadable,
            'n_gitlink_skipped': len(gitlinks),
            'n_exempted': len(exempted), 'exempted': exempted,
            'n_findings': len(findings), 'findings': findings,
            'coverage_complete': (checked + len(exempted) + len(gitlinks) == len(paths))}


def self_test(repo):
    """Prove every rule fires, using SYNTHETIC throwaway values.  No real secret is involved."""
    hex64 = 'deadbeef' * 8                       # synthetic 64-hex, obviously not a credential
    cases = [
        ('R1_execution_secret_path', 'R1', None),
        ('R2_bare_64_hex_token', 'R2', (hex64 + '\n').encode()),
        ('R2_bare_64_hex_token', 'R2q', ('"%s",\n' % hex64).encode()),
        ('R3_credential_assignment', 'R3', ('pass' + 'word = "hunter2hunter2"\n').encode()),
        # the JSON shape is the likeliest real one, and a bare leading `\b` missed it entirely
        ('R3_credential_assignment', 'R3_json',
         ('  "pass' + 'word": "hunter2hunter2"\n').encode()),
        ('R3_credential_assignment', 'R3_yaml', ('api_' + 'key: sk-live-abcdef123456\n').encode()),
        # NOTE: each fixture's ONE-LINE SOURCE TEXT must not itself read as a credential, and a
        # comment that QUOTES the shape counts.  Splitting only at the key name is not enough,
        # because the key and its value still meet inside the source line and the scanner then
        # reports its own fixture.  The split is therefore placed between the key and the colon, so
        # key and value never appear together anywhere in this file.
        ('R3_credential_assignment', 'R3_client_secret',
         ('{"client_' + 'secret' + '": "abcd1234efgh5678"}\n').encode()),
        ('R4_pem_private_key_header', 'R4',
         ('-----' + 'BEGIN RSA PRIVATE ' + 'KEY-----\n').encode()),
    ]
    results = []
    for expect, name, raw in cases:
        if expect == 'R1_execution_secret_path':
            hit = fnmatch('x/guard/execution_secret_abc.txt', 'execution_secret_*') or \
                fnmatch(Path('x/guard/execution_secret_abc.txt').name, 'execution_secret_*')
            results.append({'case': name, 'expected_rule': expect, 'detected': bool(hit)})
            continue
        hits = scan_bytes(raw)
        results.append({'case': name, 'expected_rule': expect,
                        'detected': any(h[0] == expect for h in hits),
                        'rules': sorted({h[0] for h in hits})})
    # negatives: these must NOT fire
    negatives = [
        ('benign_none', b'password = None\n'),
        ('benign_env', b'api_key = os.environ["K"]\n'),
        ('benign_placeholder', b'secret: "{{ vault_secret }}"\n'),
        ('benign_redacted', b'token = "REDACTED"\n'),
        ('benign_short_hex', b'commit = "deadbeef"\n'),
        ('benign_digest_field', b'"sha256": "f786b3c202870670226f892f5ace31aa4ff04c52087e47b3be29cc8dc443a389",\n'),
        # OUR OWN evidence shapes must never trip the scanner, or the control would block the
        # research record it exists to protect
        ('benign_secret_path_field',
         b'"secret_path": "D:\\\\x\\\\guard\\\\execution_secret_ab12.txt"\n'),
        ('benign_empty_password', b'"password": "",\n'),
        ('benign_consumption_path',
         b'"consumption_path": "research/x/l1_gates/L1_V2_EXECUTION_consumption/'
         b'authorization_consumption_293b33ab.json"\n'),
    ]
    neg_results = [{'case': n, 'fired': bool(scan_bytes(b))} for n, b in negatives]
    # NOTE the last negative: a 64-hex value inside a JSON FIELD is not a bare token line; it is
    # exactly what the redaction stubs and ledgers legitimately contain.
    return {'positive_cases': results, 'all_positive_detected': all(r['detected'] for r in results),
            'negative_cases': neg_results,
            'all_negatives_clean': all(not r['fired'] for r in neg_results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=str(REPO_DEFAULT))
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--allow', action='append', default=list(DEFAULT_ALLOW))
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    repo = Path(args.repo)
    if args.self_test:
        st = self_test(repo)
        import json as _json
        print(_json.dumps(st, indent=1))
        return 0 if (st['all_positive_detected'] and st['all_negatives_clean']) else 1
    result = scan(repo, args.allow)
    result['schema'] = 'd3_l1_staged_secret_scan'
    result['rules'] = ['R1_execution_secret_path', 'R2_bare_64_hex_token',
                       'R3_credential_assignment', 'R4_pem_private_key_header',
                       'R5_staged_blob_unreadable']
    result['allow'] = list(args.allow)
    result['values_are_never_printed'] = True
    # Fail-closed on BOTH axes: a finding refuses, and so does INCOMPLETE COVERAGE -- the scanner may
    # not issue a clean verdict over a set of files it did not all read.
    result['verdict'] = ('STAGED_CONTENT_CLEAN'
                         if (result['n_findings'] == 0 and result['coverage_complete'])
                         else 'STAGED_CONTENT_REFUSED')
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    import json as _json
    print(_json.dumps(result, indent=1))
    return 0 if result['n_findings'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
