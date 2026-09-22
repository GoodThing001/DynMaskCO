"""M0 teacher 数据集读取器（路线 B / 工作包 A）。

读取 `export_coldchain_teacher.py` 产出的三层 teacher 数据集，按 context 分组候选，并把
「当前合法候选」与「有效监督标签」拆成两个不同 mask：

  - legal_mask：候选当下可被选中（certificate feasible，或 KEEP/DEFER 伪动作）。这是在线
    决策时已知的约束，不含未来终局信息。
  - supervision_mask：候选的终局标签可用于成本监督（rolled_out 且 service_ok 且无协议错误
    且 contract 身份一致）。终局 service_ok 只能用于离线监督，绝不能作为部署时提前筛掉候选
    的依据。

无有效监督的 context 保留（记录原因与计数），训练时跳过。同源基础实例的所有事件/客户/候选
通过 context_id 同组，读取器不把候选行当独立样本随机切分。

用法：
    from coldchain_teacher_dataset import load_teacher_dataset
    ds = load_teacher_dataset(teacher_dir, data_path=<npz>)
    for ctx in ds.contexts:
        cands = ds.candidates_by_context[ctx['context_id']]
        legal = ds.legal_mask(cands)
        sup = ds.supervision_mask(cands)
"""
import hashlib
import json
import os


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


class TeacherDataset:
    """已读取并校验的 teacher 数据集。"""

    def __init__(self, manifest, contexts, candidates, data_hash=None, data_path=None):
        self.manifest = manifest
        self.contexts = contexts
        self.candidates = candidates
        self.data_hash = data_hash
        self.data_path = data_path
        self.candidates_by_context = {}
        for c in candidates:
            self.candidates_by_context.setdefault(c['context_id'], []).append(c)
        self._validate_references()
        self.unsupervised = self._collect_unsupervised()

    # ------------------------------------------------------------------ #
    # 校验
    # ------------------------------------------------------------------ #
    def _validate_references(self):
        ctx_ids = {c['context_id'] for c in self.contexts}
        missing = set(self.candidates_by_context) - ctx_ids
        if missing:
            raise ValueError(f"candidates 引用了不存在的 context：{sorted(missing)[:5]}")
        # 每个 context 至少有一个候选（含 KEEP/DEFER）。
        empty = [cid for cid in ctx_ids if not self.candidates_by_context.get(cid)]
        if empty:
            raise ValueError(f"存在无候选的 context：{empty[:5]}")

    def _collect_unsupervised(self):
        out = []
        for ctx in self.contexts:
            cands = self.candidates_by_context.get(ctx['context_id'], [])
            sup = self.supervision_mask(cands)
            if not any(sup):
                out.append({
                    'context_id': ctx['context_id'],
                    'n_candidates': len(cands),
                    'reason': 'no_service_ok_supervision',
                })
        return out

    # ------------------------------------------------------------------ #
    # mask
    # ------------------------------------------------------------------ #
    def legal_mask(self, cands):
        """当前合法候选：certificate feasible 或 KEEP/DEFER 伪动作。"""
        mask = []
        for c in cands:
            if c.get('is_pseudo'):
                mask.append(True)
            else:
                mask.append(bool(c.get('certificate', {}).get('feasible', False)))
        return mask

    def supervision_mask(self, cands):
        """有效监督标签：存在完整终局 outcome 且 service_ok 且无协议错误（可参与成本排序）。

        `rolled_out` 只记录计算来源（是否额外跑了 rollout），不作为标签资格判据：DEFER 复用
        baseline outcome（rolled_out=false），仍是有效监督。
        """
        mask = []
        for c in cands:
            has_outcome = c.get('outcome') is not None
            ok = has_outcome and bool(c.get('service_ok', False))
            ok = ok and not bool(c.get('protocol_error', False))
            mask.append(ok)
        return mask


def load_teacher_dataset(dataset_dir, data_path=None):
    """加载并校验 teacher 数据集。返回 TeacherDataset。

    data_path 提供时，核对其 SHA-256 与 manifest.dataset_hash 一致。
    """
    manifest_path = os.path.join(dataset_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"缺 manifest.json：{manifest_path}")
    if not os.path.exists(os.path.join(dataset_dir, 'COMPLETE')):
        raise ValueError("teacher 数据集缺 COMPLETE（未完成或失败）")
    if os.path.exists(os.path.join(dataset_dir, 'FAILED')):
        raise ValueError("teacher 数据集标记为 FAILED，拒绝读取")

    manifest = json.load(open(manifest_path, encoding='utf-8'))
    if manifest.get('schema') != 'o0cc-teacher-dataset-v1':
        raise ValueError(f"unsupported schema: {manifest.get('schema')}")

    contexts = [json.loads(l) for l in open(os.path.join(dataset_dir, 'contexts.jsonl'),
                                            encoding='utf-8')]
    candidates = [json.loads(l) for l in open(os.path.join(dataset_dir, 'candidates.jsonl'),
                                              encoding='utf-8')]

    data_hash = manifest.get('dataset_hash')
    if data_path is not None:
        actual = _sha256_file(data_path)
        if actual != data_hash:
            raise ValueError(f"NPZ hash 与 manifest.dataset_hash 不符："
                             f"{actual[:12]} vs {data_hash[:12]}")
    return TeacherDataset(manifest, contexts, candidates, data_hash=data_hash,
                          data_path=data_path)
