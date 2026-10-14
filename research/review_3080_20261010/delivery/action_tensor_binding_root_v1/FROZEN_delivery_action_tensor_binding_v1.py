"""Engineering action tensors bound to a temporary skeleton; no model or solver."""
import hashlib,json
import numpy as np
FIELDS=('demands','temp_class','reveal_time','ready_time','tw_start','tw_end','service_time')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def raw_nodes(data):
    coords=np.asarray(data['coords'],dtype=np.float64)
    out=np.column_stack([coords]+[np.asarray(data[k],dtype=np.float64) for k in FIELDS])
    if out.ndim!=2 or out.shape[1]!=9 or not np.isfinite(out).all():raise ValueError('raw9 node fields')
    return out
def batch(data,key,step,before,remaining,rows):
    # Input does not accept teacher routes, labels, score values or final ranks.
    if not rows or len(set(remaining))!=len(remaining):raise ValueError('empty/duplicate domain')
    actions=np.asarray([r[:5] for r in rows],dtype=np.int32)
    if actions.shape!=(len(rows),5) or len({tuple(r) for r in actions.tolist()})!=len(rows):raise ValueError('action identity')
    nodes=raw_nodes(data);features=[]
    for i,v,p,a,b in actions.tolist():
        if i not in remaining or not 0<=v<len(before) or not 0<=p<=len(before[v]):raise ValueError('action domain')
        route=before[v]
        if a!=(route[p-1] if p else 0) or b!=(route[p] if p<len(route) else 0):raise ValueError('temporary anchor identity')
        if any(not 0<=j<len(nodes) for j in (i,a,b)):raise ValueError('node bound')
        features.append(nodes[[i,a,b]])
    eligible=np.asarray([bool(r[5]) and bool(r[6]) for r in rows],dtype=np.bool_)
    tie=np.asarray([r[8] for r in rows],dtype=np.float64)
    if not np.isfinite(tie).all():raise ValueError('distance tie value')
    context=digest(dict(key=key,step=step,before=before,remaining=remaining,data=data))
    identities=digest(dict(actions=actions.tolist(),eligible=eligible.tolist(),distance_hex=[float(v).hex() for v in tie]))
    return dict(actions=actions,node_features=np.asarray(features,dtype=np.float32),eligible=eligible,distance_tie=tie,context_sha256=context,action_sha256=identities)
def targets(before,actions,teacher_routes):
    owner={j:v for v,route in enumerate(teacher_routes) for j in route};rank={j:p for route in teacher_routes for p,j in enumerate(route)}
    if len(owner)!=sum(map(len,teacher_routes)):raise ValueError('teacher coverage duplicate')
    return np.asarray([float(owner[i]==v and all(rank[j]<rank[i] for j in before[v][:p]) and all(rank[j]>rank[i] for j in before[v][p:])) for i,v,p,a,b in actions.tolist()],dtype=np.float32)
def score_packet(b,values,source):
    values=np.asarray(values)
    if source not in ('observed_ideal_fixture','unqualified_numeric_fixture'):raise ValueError('unqualified source kind')
    if values.dtype!=np.float32 or values.shape!=(len(b['actions']),) or not np.isfinite(values).all():raise ValueError('finite float32 score vector')
    return dict(context_sha256=b['context_sha256'],action_sha256=b['action_sha256'],values=values.copy(),declared_source=source,learned_output_qualified=False)
def consume(b,packet):
    if packet['context_sha256']!=b['context_sha256'] or packet['action_sha256']!=b['action_sha256'] or packet['declared_source'] not in ('observed_ideal_fixture','unqualified_numeric_fixture') or packet['learned_output_qualified']:raise ValueError('score source/context identity')
    values=packet['values']
    if not isinstance(values,np.ndarray) or values.dtype!=np.float32 or values.shape!=(len(b['actions']),) or not np.isfinite(values).all():raise ValueError('score vector identity')
    return values.copy()
