"""Append the standing git-delivery convention to the maskco automation prompt.

The user's 2026-10-14 standing instruction: after every completed piece of work, push to
https://github.com/GoodThing001/DynMaskCO .  It must apply to autonomous heartbeat rounds
too, so it is written into the automation prompt, not only into the repo docs.

The prompt is a TOML basic string (single line, \\n escapes, no internal double quotes),
so the appended text must contain no double quote and no backslash.
"""
import shutil
import sys
from pathlib import Path

CFG = Path.home() / '.codex' / 'automations' / 'maskco' / 'automation.toml'
ANCHOR = '仅实质阶段完成、异常、正式裁决或需人类科学决定时通知。"'

ADDITION = (
    '每完成一次工作（一个可交付单元完成、工程阶段闭环、或诊断/实验出结果）**立即**用 '
    'python tools/git_deliver.py --message-file <msg.txt> 提交并推送，'
    '两个分支都要同步：codex/o0cc-cal-phys 与 main（只做快进，origin/main 必须是 HEAD 的祖先）。'
    '推送前核验身份哈希、排除无 .gitmodules 映射的幽灵子模块 CC_Compare/MAPT_code、'
    '对已登记 raw sha256 的字节容器根做 git add --renormalize；'
    '推送后校验 origin 两个分支都等于本地 HEAD。'
    '若推送失败、检测到并发写入者、或 origin/main 不能快进，如实记录并报告，**不得静默跳过**推送，'
    '也**不得** force push 或改写已发布历史。'
)


def main():
    text = CFG.read_text(encoding='utf-8')
    if 'git_deliver.py' in text:
        print('already present; nothing to do')
        return 0
    if ANCHOR not in text:
        print('ANCHOR NOT FOUND -- refusing to edit blindly')
        return 1
    if '"' in ADDITION or '\\' in ADDITION:
        print('ADDITION contains a quote or backslash; refusing (TOML basic string)')
        return 1
    shutil.copy2(CFG, str(CFG) + '.bak-20261014')
    new = text.replace(ANCHOR, ANCHOR[:-1] + ADDITION + '"', 1)
    assert new != text and new.count('prompt = "') == 1
    CFG.write_text(new, encoding='utf-8', newline='\n')
    print('patched %s' % CFG)
    # sanity: the file must still parse as TOML
    try:
        import tomllib
        with CFG.open('rb') as f:
            d = tomllib.load(f)
        print('toml parses OK; prompt length =', len(d['prompt']), '; status =', d['status'])
        assert 'git_deliver.py' in d['prompt']
    except Exception as exc:                                   # pragma: no cover
        shutil.copy2(str(CFG) + '.bak-20261014', CFG)
        print('TOML PARSE FAILED (%r) -- restored from backup' % exc)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
