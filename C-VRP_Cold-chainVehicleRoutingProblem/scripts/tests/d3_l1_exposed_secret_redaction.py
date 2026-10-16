#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""EXPOSED-SECRET SANITISER: archive the originals privately, redact the PUBLIC copies.

WHAT HAPPENED
  Two 256-bit execution secrets were committed and pushed to the public GitHub repository
  `GoodThing001/DynMaskCO` as part of the raw run directories.  They are single-use secrets minted per
  run by `secrets.token_hex(32)` and validated by `d3_l1_v120_execution_context.compute_hmac`; each
  run's credential was consumed one-shot BEFORE the secret was published, so neither secret can
  authorise any new work.  They are nevertheless published key material and are handled as EXPOSED.

WHAT THIS TOOL DOES
  1. copies the ORIGINAL bytes to a private archive OUTSIDE the repository (default `~/dynmaskco_private`),
     so the complete research record survives intact and off the public internet;
  2. records, for each secret, the file sha256 and the secret's own sha256 -- these are ONE-WAY
     digests, so a reader can prove WHICH secret was exposed without being handed its value;
  3. overwrites the in-repo files with a redaction stub that states the same facts and nothing else.

WHY NOT `git filter-repo` HERE
  Rewriting history changes every downstream commit SHA, which invalidates the signed execution
  authorisation that names `git HEAD 558fb5c...` and the whole commit-pinned audit chain.  The purging
  runbook is emitted SEPARATELY so that that cost is a deliberate, explicitly-taken decision.

This tool NEVER prints a secret value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
EV = (REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1')

# (in-repo path, run_id, when the secret was minted/consumed, the commit that published it)
TARGETS = (
    {
        'rel': 'l1_v2_execution_run/guard/execution_secret_12adf7457768bfb5.txt',
        'run_id': 'd3l1-l1-two-arm-20261016-l1v2',
        'published_in_commit': 'bdbc18ad7',
        'consumed_utc': '2026-10-16T03:24:12Z',
        'note': 'the L1-v2 formal comparison run; its credential was consumed one-shot before publication',
    },
    {
        'rel': 'v122_evidence/run_v122/guard/execution_secret_b3e36be45ceb6bb4.txt',
        'run_id': 'd3l1-real-f-forward-20261015-v122',
        'published_in_commit': 'PREDATES the L1-v2 round (already public before this review)',
        'consumed_utc': '2026-10-15',
        'note': 'the V1.22 real-F forward run; PRE-EXISTING exposure, found during this review',
    },
)
STUB_TEMPLATE = """REDACTED-EXPOSED-EXECUTION-SECRET
==================================
The 256-bit execution secret that was written to this path has been REMOVED from the public
repository because it was published here in error.

  run_id                  : {run_id}
  secret file sha256      : {file_sha256}
  secret value sha256     : {secret_sha256}   (one-way; the value itself is NOT recorded here)
  secret length           : {n_bytes} bytes of lowercase hex (256 bits)
  published_by_commit     : {published_in_commit}
  credential_consumed_utc : {consumed_utc}
  redacted_utc            : {redacted_utc}

DISPOSITION: the secret is treated as EXPOSED and is NEVER reused.  Each run mints its own secret
with `secrets.token_hex(32)` at a freshly named path, so this value carries no residual authority:
the credential it guarded was consumed one-shot before publication, and no future run reads it.

WHAT IS PRESERVED, AND WHERE
  The ORIGINAL bytes are preserved byte-for-byte in a PRIVATE archive outside this repository:
      {archive_path}
  The research record is therefore intact; only the public copy is redacted.  The sha256 values above
  are the same ones a verifier would compute from the archived original, so the identity of the
  exposed secret remains checkable without republishing it.

  The `EXECUTION_CONTEXT_*.json` record that names this path and carries the HMAC computed from this
  secret is LEFT INTACT: an HMAC is not reversible to its key, the key is already published, and that
  record is load-bearing evidence that the guard bound run, directory, consumed credential and nonce.
{extra}
"""


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive-root', default=str(Path.home() / 'dynmaskco_private'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    archive_root = Path(args.archive_root).expanduser().resolve()
    stamp = '2026-10-16'
    rows = []
    for t in TARGETS:
        src = EV / t['rel']
        row = {'rel': t['rel'], 'run_id': t['run_id'], 'in_repo_exists': src.is_file()}
        if not src.is_file():
            rows.append(row)
            continue
        original = src.read_bytes()
        row['n_bytes'] = len(original)
        row['file_sha256'] = sha_bytes(original)
        row['secret_sha256'] = sha_bytes(original.strip())
        row['already_redacted'] = original.startswith(b'REDACTED-EXPOSED-EXECUTION-SECRET')
        if row['already_redacted']:
            rows.append(row)
            continue
        arch_dir = archive_root / 'exposed_execution_secrets' / stamp
        arch_file = arch_dir / (t['run_id'] + '__' + Path(t['rel']).name)
        row['archive_path'] = str(arch_file)
        if not args.dry_run:
            arch_dir.mkdir(parents=True, exist_ok=True)
            if not arch_file.exists():
                shutil.copy2(src, arch_file)
            # a private manifest so the archive explains itself without the repository
            (arch_dir / 'README.txt').write_text(
                'EXPOSED EXECUTION SECRETS - PRIVATE ARCHIVE, DO NOT PUBLISH\n'
                '=========================================================\n'
                'These are the ORIGINAL bytes of 256-bit execution secrets that were published in the\n'
                'public GitHub repository GoodThing001/DynMaskCO.  They are preserved here so the\n'
                'research record stays complete and off the public internet.\n\n'
                'They are EXPOSED and must NEVER be reused.  Each run mints its own secret, and every\n'
                'credential these guarded was consumed one-shot before publication, so they carry no\n'
                'residual authority.\n\n'
                'The public repository keeps a redaction stub recording each file sha256 and secret\n'
                'sha256 (one-way), so the identity of what was exposed stays checkable without\n'
                'republishing the values.\n', encoding='utf-8')
            extra = ('\n  A second copy of the {kind} exposure context is:\n      {ctx}\n'.format(
                kind=t['run_id'], ctx=archive_root))
            stub = STUB_TEMPLATE.format(
                run_id=t['run_id'], file_sha256=row['file_sha256'],
                secret_sha256=row['secret_sha256'], n_bytes=row['n_bytes'],
                published_in_commit=t['published_in_commit'], consumed_utc=t['consumed_utc'],
                redacted_utc=stamp, archive_path=str(arch_file), extra='')
            src.write_text(stub, encoding='utf-8', newline='\n')
            row['stub_sha256'] = sha_bytes(src.read_bytes())
            row['redacted'] = True
        else:
            row['redacted'] = False
        rows.append(row)
    payload = {
        'schema': 'd3_l1_exposed_secret_redaction',
        'n_targets': len(rows),
        'n_redacted': sum(1 for r in rows if r.get('redacted')),
        'archive_root': str(archive_root),
        'rows': rows,
        'rule': 'never print, echo or commit a secret value; record only one-way digests',
    }
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2, sort_keys=True, default=repr),
                                   encoding='utf-8')
    print(json.dumps(payload, indent=2, default=repr))
    return 0


if __name__ == '__main__':
    sys.exit(main())
