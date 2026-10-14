"""Exclude only the oversized raw-evidence dumps (json/jsonl >= 1 MiB) from git.

All source (.py), documentation (.md) and every artifact below 1 MiB stay tracked.
The excluded files remain untouched on disk; git simply does not track them.
"""
import subprocess
from pathlib import Path

ROOT = Path(r'D:\PyCharm_\MASKCO-Main')
THRESHOLD = 1 * 1024 * 1024
MARK_BEGIN = '# ==== BEGIN generated: oversized raw-evidence exclusions ===='
MARK_END = '# ==== END generated ===='


def run(*args):
    return subprocess.run(['git', *args], cwd=str(ROOT), capture_output=True, text=True,
                            encoding='utf-8', errors='replace')


def main():
    run('reset')
    st = run('status', '--porcelain=v1', '-z', '--untracked-files=all')
    entries = [e for e in st.stdout.split('\0') if e.strip()]
    paths = []
    for e in entries:
        code, p = e[:2], e[3:]
        if code.strip() == 'D':
            continue
        paths.append(p)

    big = []
    for p in paths:
        f = ROOT / p
        try:
            if f.is_file() and f.stat().st_size >= THRESHOLD and f.suffix.lower() in ('.json', '.jsonl'):
                big.append(p)
        except OSError:
            pass
    big.sort()

    gi = ROOT / '.gitignore'
    text = gi.read_text(encoding='utf-8')
    if MARK_BEGIN in text:
        head = text.split(MARK_BEGIN)[0].rstrip('\n')
    else:
        head = text.rstrip('\n')
    block = [MARK_BEGIN,
             '# Reason: the working tree holds ~5.6 GB of raw audit/evidence dumps; GitHub rejects',
             '# files over 100 MB and the existing tracked tree is 383 MB. Code (.py), docs (.md) and',
             '# every artifact under 1 MiB remain tracked. The excluded files stay on disk.',
             '# Generated from the staged set; regenerate when new large evidence is added.']
    block += ['/' + p for p in big]
    block.append(MARK_END)
    gi.write_text(head + '\n\n' + '\n'.join(block) + '\n', encoding='utf-8', newline='\n')
    print('excluded entries:', len(big))
    print('excluded MB: %.1f' % (sum((ROOT / p).stat().st_size for p in big) / 1024 / 1024))
    for p in big[:5]:
        print('  e.g.', p)


if __name__ == '__main__':
    main()
