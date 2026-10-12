"""Deterministic score-quantile sampling and source-condition objectives."""
import math
import numpy as np

def normalize_scores(scores):
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or len(values)<2:
        raise ValueError('Finite candidate list required')
    span=float(values.max()-values.min())
    return (values-values.min())/span if span>1e-12 else np.zeros_like(values)

def sample_candidates(scores,positive,condition_negative=None,k=8,policy='stratified'):
    if not 2<=k<=len(scores) or not 0<=positive<len(scores):raise ValueError('Invalid bounded sample')
    normalized=normalize_scores(scores)
    selected=[positive]
    if condition_negative is not None:
        if not 0<=condition_negative<len(scores) or condition_negative==positive:raise ValueError('Invalid condition conflict')
        selected.append(condition_negative)
    available=[i for i in range(len(scores)) if i not in selected]
    remaining=k-len(selected)
    if policy=='hard_top':
        selected+=sorted(available,key=lambda i:(-normalized[i],i))[:remaining]
    elif policy=='stratified':
        anchors=np.quantile(normalized[available],np.linspace(0,1,remaining))
        for anchor in anchors:
            index=min((i for i in available if i not in selected),key=lambda i:(abs(normalized[i]-anchor),i))
            selected.append(index)
    else:raise ValueError('Unknown sampling policy')
    if len(set(selected))!=k:raise AssertionError('Nonunique sample')
    values=normalized[selected]
    histogram=np.histogram(values,bins=np.linspace(0,1,9))[0].astype(float)/k
    return selected,{'policy':policy,'normalized_range':float(values.max()-values.min()),
                    'binned_entropy':float(-sum(p*math.log(p) for p in histogram if p>0)),
                    'standard_deviation':float(values.std()),'constant_scores':bool(np.ptp(normalized)==0),
                    'forced_positive':positive,'forced_source_condition_conflict':condition_negative}

def ranking_loss(student,teacher,positive_index,condition_index=None,distill=True):
    import torch
    import torch.nn.functional as F
    # Keep reviewed source supervision even when the teacher prefers a wrong role.
    hard=F.cross_entropy(student,positive_index)
    kl=F.kl_div(F.log_softmax(student.float(),dim=-1),F.softmax(teacher.float(),dim=-1),reduction='batchmean')
    conflict=student.sum()*0
    if condition_index is not None:
        mask=condition_index>=0
        if mask.any():
            pos=student[mask,positive_index[mask]]
            neg=student[mask,condition_index[mask]]
            conflict=F.softplus(neg-pos).mean()
    total=hard+conflict+(kl if distill else 0)
    return total,{'source_ce':hard,'condition_pairwise':conflict,'listwise_kl':kl}
