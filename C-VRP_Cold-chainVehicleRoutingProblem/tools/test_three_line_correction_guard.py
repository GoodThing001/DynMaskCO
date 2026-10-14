"""Synthetic filesystem tests; no model or scientific experiment executes."""
import hashlib
import tempfile
from pathlib import Path
from verify_three_line_corrections import protected_check

with tempfile.TemporaryDirectory() as d:
    root = Path(d) / 'extension'
    root.mkdir()
    p = root / 'sample'
    p.write_bytes(b'original')
    h = hashlib.sha256(b'original').hexdigest()
    assert protected_check(root, {'sample': h}) == []
    p.write_bytes(b'changed')
    assert protected_check(root, {'sample': h})[0]['reason'] == 'hash_mismatch'
    assert protected_check(root, {'missing': h})[0]['reason'] == 'missing'
    assert protected_check(root, {'../arbitrary': h})[0]['reason'] == 'outside_root'
    q = root.parent / 'MASKCO_code/ckpts/cvrp100.ckpt'
    q.parent.mkdir(parents=True)
    q.write_bytes(b'original')
    assert protected_check(root, {'../MASKCO_code/ckpts/cvrp100.ckpt': h}) == []
print('PASS: unchanged, changed, missing, traversal rejection, explicit read-only upstream pin')
