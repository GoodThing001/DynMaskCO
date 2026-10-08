#!/usr/bin/env bash
# Derive a new immutable source snapshot that runs only the cond_hist arm.
# The scoring, dataset, contract and identity modules remain byte-identical.
set -euo pipefail
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem

PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
BASE=results/a3_baseline_src_frozen_20261007
DST=results/a3_baseline_condonly_v2_src_frozen_20261007
test -f "$BASE/PROVENANCE.json"
if test -e "$DST"; then
  if test "${1:-}" != --resume || grep -q -- '--only-cond' \
      "$DST/scripts/evaluation/run_a1_step2_gate.py"; then
    echo "REJECT: destination exists or already modified: $DST"
    exit 2
  fi
  echo "RESUME: completing unmodified copy at $DST"
elif test "${1:-}" = --resume; then
  echo 'REJECT: --resume requires an incomplete copy'
  exit 2
fi
(cd "$BASE" && sha256sum -c SOURCE_MANIFEST.sha256) >/dev/null
if test "${1:-}" != --resume; then cp -a "$BASE" "$DST"; fi
chmod u+w "$DST" "$DST/scripts/evaluation/run_a1_step2_gate.py" \
  "$DST/SOURCE_MANIFEST.sha256" "$DST/PROVENANCE.json"

"$PY" - "$BASE" "$DST" <<'PY'
import hashlib
import json
import pathlib
import sys

base, dst = map(pathlib.Path, sys.argv[1:])
p = dst / 'scripts/evaluation/run_a1_step2_gate.py'
s = p.read_text(encoding='utf-8')
needle_arg = '    ap.add_argument("--workers", type=int, default=1,'
needle_filter = '    penalty_items = ([(args.penalty, PENALTIES[args.penalty])]'
needle_report = '    for pname in results:\n        cond_rows = per_day["cond_hist"][pname]'
if any(s.count(needle) != 1 for needle in (needle_arg, needle_filter, needle_report)):
    raise SystemExit('REJECT: unexpected rr3 driver structure')
s = s.replace(needle_arg,
              '    ap.add_argument("--only-cond", action="store_true",\n'
              '                    help="Run cond_hist only; no shared scoring change")\n'
              + needle_arg, 1)
s = s.replace(needle_filter,
              '    if args.only_cond:\n'
              '        samplers = {"cond_hist": samplers["cond_hist"]}\n\n'
              + needle_filter, 1)
s = s.replace(needle_report,
              '    for pname in results:\n'
              '        if args.only_cond:\n'
              '            results[pname]["per_day"] = {"cond_hist": per_day["cond_hist"][pname]}\n'
              '            continue  # No cond-uncond claim in a baseline-only run.\n'
              '        cond_rows = per_day["cond_hist"][pname]', 1)
s = s.replace('"note": "A-v1',
              '"baseline_only": bool(args.only_cond),\n        "note": "A-v1', 1)
p.write_text(s, encoding='utf-8')

manifest = {}
for line in (base / 'SOURCE_MANIFEST.sha256').read_text().splitlines():
    if line and not line.startswith('#'):
        digest, rel = line.split(None, 1)
        manifest[rel.strip()] = digest
for rel, old_digest in manifest.items():
    digest = hashlib.sha256((dst / rel).read_bytes()).hexdigest()
    if rel != 'scripts/evaluation/run_a1_step2_gate.py' and digest != old_digest:
        raise SystemExit('REJECT: unexpected changed file: ' + rel)
    manifest[rel] = digest
(dst / 'SOURCE_MANIFEST.sha256').write_text(
    '# A3 cond-only baseline source; rr3 driver arm filter only\n' +
    ''.join('%s  %s\n' % (manifest[rel], rel) for rel in sorted(manifest)),
    encoding='utf-8')
provenance = json.loads((base / 'PROVENANCE.json').read_text(encoding='utf-8'))
provenance['derived_from'] = str(base)
provenance['driver_change'] = '--only-cond filters tasks and omits unavailable cross-arm report; shared SAA untouched'
provenance['step2_driver_sha256'] = manifest['scripts/evaluation/run_a1_step2_gate.py']
(dst / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2),
                                     encoding='utf-8')
print('cond-only frozen manifest: %d files' % len(manifest))
PY

(cd "$DST" && sha256sum -c SOURCE_MANIFEST.sha256)
chmod -R a-w "$DST"
echo "SEALED: $DST"
