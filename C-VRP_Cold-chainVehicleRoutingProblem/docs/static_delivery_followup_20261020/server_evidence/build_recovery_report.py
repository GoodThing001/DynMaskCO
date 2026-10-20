"""Build the bounded remote-recovery report from saved read-only discovery evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
EXT=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
source=EXT/'data/public_sources_20261020/refrigerated_routing_rrp'
reference=source/'Solutions/D1/sol-D1-n9-np32.txt'
raw=reference.read_bytes()
sha=hashlib.sha256(raw).hexdigest()
manifest=json.loads((source/'download_manifest.json').read_text(encoding='utf-8'))
entry=next(x for x in manifest if x['file']=='Solutions/D1/sol-D1-n9-np32.txt')
assert entry['sha256']==sha
lines=raw.decode('utf-8').splitlines()
route=lines[lines.index('Route: ')+1].split()
wait=lines[lines.index('Waiting times: ')+1].split()
assert len(route)==9 and len(wait)==8
linux=json.loads((HERE/'linux_discovery.json').read_text(encoding='utf-8'))
windows=json.loads((HERE/'windows_discovery.json').read_text(encoding='utf-8'))
assert not linux['stopped_early'] and not windows['stopped_early']
assert not linux['errors'] and not windows['errors']
copies=linux['known_uploaded_copy_checks']
assert len(copies)==5 and all(x.get('matches_local') is True for x in copies)
evidence={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir()) if p.is_file()}
result={
 'created_at_utc':datetime.now(timezone.utc).isoformat(),
 'scope':'Read-only bounded discovery in configured Linux and Windows user-owned MASKCO projects only',
 'finding':'No independent older RRP implementation or missing-reference replacement found within the searched scope',
 'source_commit':entry['commit'],
 'local_reference':{'path':str(reference.relative_to(ROOT)).replace('\\','/'),'bytes':len(raw),'sha256':sha,'source_manifest_matches':True,'route_values':len(route),'waiting_values':len(wait),'missing_waiting_value_recovered':False},
 'linux':linux,'windows':windows,
 'known_uploaded_copy_checks':copies,
 'downloaded_recovery_files':[],
 'remote_mutations':False,'training_started':False,
 'limitations':['Search excludes private/credential/secret paths, repository internals, dependency caches, checkpoints, symlinks and Windows reparse points.','No server paths outside the two configured user-owned project roots were scanned.','Archives were not unpacked or searched internally; files whose names and parent paths lack RRP/refrigerated/energies13_6214 markers may not be detected.','run01-run05 are known uploads of this local source; identical hashes do not independently resolve the missing wait.'],
 'evidence_sha256':evidence,
}
(HERE.parent/'server_recovery.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
report=f'''# RRP服务器缺项查找证据（2026-10-20）

在两台已授权服务器的用户项目范围内，未找到独立的历史RRP实现，也未找到可补全`sol-D1-n9-np32.txt`缺项的独立参考副本。此结论仅覆盖下列搜索范围；不表示其他位置绝无材料。未修改服务器文件、未启动训练、未删除文件。

## 查找范围与结果

| 服务器 | 根目录 | 遍历条目/目录 | RRP相关命中 |
|---|---|---|---|
| Linux | `{linux['root']}` | {linux['visited_entries']}/{linux['visited_directories']} | 仅本次`research/static_rrp_20261020`目录 |
| Windows 3080 | `{windows['root']}` | {windows['visited_entries']}/{windows['visited_directories']} | 0 |

先用SFTP列出配置所指项目根，再作只读、有界文件名遍历。每台设深度12、18万条目、22秒上限；两台均自然完成，未触及时间/条目上限、深度剪枝均为0、遍历错误均为0。匹配文件/目录名中的`rrp`、`refrigerated`、`energies.?13.?6214`、`sol-D1-n9-np32`，并收集相关目录内的JSON/TXT/C++/MiniZinc/Python/Markdown/ZIP条目。没有扫描其他用户目录，没有读出凭据。

排除`.git`、缓存、依赖目录、checkpoints、private及名称含private/credential/secret的路径；Linux不跟随符号链接，Windows拒绝reparse/junction。本次Linux上传的run01–run05从发现遍历排除。未展开压缩包，且无相关命名线索的实现文件可能不被发现。

## 已知上传副本的SHA对照

额外仅对run01–run05的命名D1-n9文件只读计算SHA，五份均为{len(raw)}字节，且均与本地原TXT一致：

`{sha}`

服务器完整路径逐份保存在[机器记录](server_recovery.json)的`known_uploaded_copy_checks`。这些是本次从本地上传的同源副本，不能作为独立恢复来源；没有把其数量当作正确性证据。未下载任何恢复文件。

## 缺项维持原状

本地文件为`data/public_sources_20261020/refrigerated_routing_rrp/Solutions/D1/sol-D1-n9-np32.txt`，作者固定commit为`{entry['commit']}`，当前SHA与原下载清单匹配。原TXT的Route包含{len(route)}个值，Waiting times仅{len(wait)}个值。服务器搜索未恢复第9个等待值，不能据全零外观推定缺项也是0；保持原字节与不完整标记，不据此声称全部参考成本已复现。

原始查找证据：[Linux](server_evidence/linux_discovery.json)、[Windows](server_evidence/windows_discovery.json)。机器报告保存搜索参数、逐份副本SHA、局限及证据文件SHA；报告仅记录证据所需项目路径，不输出连接主机、认证字段或配置内容。
'''
(HERE.parent/'server_recovery.md').write_text(report,encoding='utf-8')
print(json.dumps({'report':str(HERE.parent/'server_recovery.md'),'machine':str(HERE.parent/'server_recovery.json'),'local_sha256':sha,'five_uploaded_copies_match':True,'independent_recovery_found':False},ensure_ascii=False))
