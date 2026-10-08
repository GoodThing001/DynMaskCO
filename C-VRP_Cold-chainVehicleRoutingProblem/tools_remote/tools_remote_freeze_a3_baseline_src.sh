#!/usr/bin/env bash
# Seal a source-compatible cond_hist baseline for the completed A3 batches.
# The rr3 step-2 driver keeps the original four-key shared_config schema;
# the A3 shared SAA source is copied only after its recorded hash is verified.
set -euo pipefail
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem

PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
REF=results/a1_a3_online_density_B_20260926_40_1003_040454/gate.json
BASE=results/rr3_src_frozen
DST=results/a3_baseline_src_frozen_20261007

test -f "$REF"
test -f "$BASE/SOURCE_MANIFEST.sha256"
if test -e "$DST"; then
  if test "${1:-}" != --resume || test -e "$DST/PROVENANCE.json"; then
    echo "REJECT: destination already exists: $DST (only an incomplete copy may use --resume)"
    exit 2
  fi
  echo "RESUME: completing the incomplete immutable copy at $DST"
else
  if test "${1:-}" = --resume; then
    echo "REJECT: --resume requires an incomplete destination"
    exit 2
  fi
fi

# Fail before copying if any of the 12 A3 source files have changed.
"$PY" - "$REF" <<'PY'
import hashlib
import json
import pathlib
import sys

doc = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
expected = doc['results']['20260926']['_identity']['source_sha256']
bad = []
for rel, digest in sorted(expected.items()):
    path = pathlib.Path(rel)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        bad.append(rel)
if bad:
    raise SystemExit('REJECT: A3 source mismatch: ' + ', '.join(bad))
print('A3 source check: %d/%d match' % (len(expected), len(expected)))
PY

if test "${1:-}" != --resume; then
  cp -a "$BASE" "$DST"
  chmod u+w "$DST/scripts/evaluation/scenario_saa.py"
  cp scripts/evaluation/scenario_saa.py "$DST/scripts/evaluation/scenario_saa.py"
fi
chmod u+w "$DST" "$DST/SOURCE_MANIFEST.sha256"

# Keep the old step-2 driver and all other rr3 modules unchanged, with the
# A3 scenario_saa.py as the only intentional replacement.
"$PY" - "$BASE" "$DST" "$REF" <<'PY'
import hashlib
import json
import pathlib
import sys

base, dst, ref = map(pathlib.Path, sys.argv[1:])
doc = json.loads(ref.read_text(encoding='utf-8'))
a3 = doc['results']['20260926']['_identity']
manifest = {}
for line in (base / 'SOURCE_MANIFEST.sha256').read_text().splitlines():
    if line and not line.startswith('#'):
        digest, rel = line.split(None, 1)
        manifest[rel.strip()] = digest
for rel, old_hash in manifest.items():
    path = dst / rel
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if rel != 'scripts/evaluation/scenario_saa.py' and digest != old_hash:
        raise SystemExit('REJECT: rr3 base changed: ' + rel)
    manifest[rel] = digest
for rel, digest in a3['shared_source_sha256'].items():
    if manifest.get(rel) != digest:
        raise SystemExit('REJECT: A3 shared source mismatch: ' + rel)
(dst / 'SOURCE_MANIFEST.sha256').write_text(
    '# A3-compatible cond_hist baseline source\n' +
    ''.join('%s  %s\n' % (manifest[rel], rel) for rel in sorted(manifest)),
    encoding='utf-8')
(dst / 'PROVENANCE.json').write_text(json.dumps({
    'purpose': 'A3 same-shared-source cond_hist baseline',
    'a3_reference_gate': str(ref),
    'rr3_driver_source': str(base),
    'a3_source_sha256': a3['source_sha256'],
    'frozen_shared_source_sha256': a3['shared_source_sha256'],
    'step2_driver_sha256': manifest['scripts/evaluation/run_a1_step2_gate.py'],
}, indent=2), encoding='utf-8')
print('frozen manifest: %d files' % len(manifest))
PY

(cd "$DST" && sha256sum -c SOURCE_MANIFEST.sha256)
chmod -R a-w "$DST"
echo "SEALED: $DST"
