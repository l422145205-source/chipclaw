"""Dev-fitted expected-gain routing on actual full-chunk retrieval candidates."""
import hashlib,json,re,time
from pathlib import Path
import numpy as np
from .config import ROOT
from .cascade_v3 import V3Student,ordered,cascade_order
from .teacher_v3 import QwenTeacher,REVISION
FEATURES=['top_gap','normalized_entropy','score_std','chinese_character_fraction','log_query_chars','top6_page_diversity']
def language(question):return 'zh' if re.search(r'[\u4e00-\u9fff]',question) else 'en'
def features(question,scores,pages):
    values=np.asarray(scores,dtype=float)
    if values.ndim!=1 or not len(values) or len(values)!=len(pages) or not np.isfinite(values).all():raise ValueError('Finite aligned candidate scores required')
    order=np.argsort(-values,kind='stable');center=values-values.max();probs=np.exp(center)/np.exp(center).sum()
    return np.array([values[order[0]]-values[order[1]] if len(values)>1 else 0,
        -float(np.sum(probs*np.log(probs+1e-12)))/max(1,np.log(len(values))),float(values.std()),
        len(re.findall(r'[\u4e00-\u9fff]',question))/max(1,len(question)),np.log1p(len(question)),
        len({pages[i] for i in order[:6]})/min(6,len(values))],dtype=float)
def expected_gain(model,vector):
    x=(np.asarray(vector)-np.asarray(model['mean']))/np.asarray(model['scale'])
    return float(np.r_[1.,x]@np.asarray(model['coefficients']))
def fit_gain(vectors,gains):
    x=np.asarray(vectors,dtype=float);y=np.asarray(gains,dtype=float)
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-6]=1.
    design=np.c_[np.ones(len(x)),(x-mean)/scale];penalty=np.diag([0.]+[10.]*x.shape[1])
    coefficients=np.linalg.solve(design.T@design+penalty,design.T@y)
    return {'mean':mean.tolist(),'scale':scale.tolist(),'coefficients':coefficients.tolist(),'ridge_lambda':10.}
class MultilingualStudent(V3Student):
    def __init__(self,path=None,device='cuda:0'):
        if path is None:
            selected=json.loads((ROOT/'reports/multilingual-v4/selected_model.json').read_text())
            if selected['selection_split']!='dev':raise ValueError('Invalid student selection split')
            path=ROOT/'.cache/v4/student'/selected['arm']/'best'
        super().__init__(path,device)
class AdaptiveRanker:
    def __init__(self,student=None,teacher=None,gate_path=None):
        self.student=student or MultilingualStudent();self.teacher=teacher
        default=ROOT/'reports/adaptive-domain-v4/gate.json'
        if not default.exists():default=ROOT/'reports/adaptive-v4/gate.json'
        self.gate=json.loads(Path(gate_path or default).read_text())
        if self.gate['student_weight_sha256']!=self.student.weight_sha or self.gate['teacher_revision']!=REVISION:raise ValueError('Gate/model identity mismatch')
        if self.gate['selection_split']!='dev' or self.gate['features']!=FEATURES:raise ValueError('Gate protocol mismatch')
        with (ROOT/'data/corpus-v3/index/chunks.jsonl').open('rb') as stream:corpus=hashlib.file_digest(stream,'sha256').hexdigest()
        if self.gate['corpus_sha256']!=corpus:raise ValueError('Gate/corpus identity mismatch')
        if self.gate['teacher_top_n']!=6:raise ValueError('Unsupported teacher shortlist')
        for model in self.gate['languages'].values():
            if len(model['coefficients'])!=7 or len(model['scale'])!=6 or len(model['mean'])!=6:raise ValueError('Gate feature dimension mismatch')
            if not np.isfinite(model['coefficients']+model['scale']+model['mean']+[model['threshold']]).all() or min(model['scale'])<=0:raise ValueError('Nonfinite or invalid gate')
        if self.teacher is not None:self._validate_teacher()
        self.last_metrics=None
    def _validate_teacher(self):
        expected=self.gate['teacher_identity'];actual=self.teacher.identity()
        for key in ['revision','weight_sha256','prompt_sha256','max_input_tokens']:
            if expected[key]!=actual[key]:raise ValueError('Gate/teacher input identity mismatch')
        if expected.get('adapter_sha256')!=actual.get('adapter_sha256'):raise ValueError('Gate/adapter identity mismatch')
    def rank(self,question,ids,texts,pages,mode='adaptive_v4'):
        if not ids or len(set(ids))!=len(ids) or not len(ids)==len(texts)==len(pages):raise ValueError('Unique aligned candidates required')
        if mode not in {'adaptive_v4','multilingual_v4','domain_teacher_v4'}:raise ValueError('Unknown v4 ranking mode')
        if mode=='domain_teacher_v4':
            if self.gate['teacher_identity'].get('kind')!='domain_lora_v4':raise ValueError('Domain teacher gate/adapter is not ready')
            if self.teacher is None:
                from .domain_teacher_v4 import DomainTeacher
                self.teacher=DomainTeacher();self._validate_teacher()
            start=time.perf_counter();scores=self.teacher.predict([(question,p) for p in texts])
            rank=ordered(scores,ids)
            self.last_metrics={'mode':mode,'language':language(question),'teacher_pairs':len(ids),
                'teacher_forward_batches':self.teacher.last_metrics['forward_batches'],'student_forward_batches':0,
                'candidate_pairs':len(ids),'latency_ms':(time.perf_counter()-start)*1000,
                'teacher_identity':self.teacher.identity(),'gate_split':'dev','escalated':True}
            return rank
        start=time.perf_counter();scores=self.student.predict([(question,p) for p in texts]);rank=ordered(scores,ids)
        lang=language(question);gate=self.gate['languages'][lang];predicted=expected_gain(gate,features(question,scores,pages))
        escalate=mode=='adaptive_v4' and gate['enabled'] and len(ids)>1 and predicted>=gate['threshold'];pairs=0;batches=0
        if escalate:
            if self.teacher is None:
                if self.gate['teacher_identity'].get('kind')=='domain_lora_v4':
                    from .domain_teacher_v4 import DomainTeacher
                    self.teacher=DomainTeacher()
                else:self.teacher=QwenTeacher()
                self._validate_teacher()
            head=rank[:6];ts=self.teacher.predict([(question,texts[i]) for i in head]);rank=cascade_order(scores,ids,ts,6)
            pairs=len(head);batches=self.teacher.last_metrics['forward_batches']
        self.last_metrics={'mode':mode,'language':lang,'predicted_page_rr_gain':predicted,'escalated':bool(escalate),
            'teacher_pairs':pairs,'teacher_forward_batches':batches,'student_forward_batches':self.student.last_metrics['forward_batches'],
            'candidate_pairs':len(ids),'latency_ms':(time.perf_counter()-start)*1000,'gate_split':'dev',
            'student_weight_sha256':self.student.weight_sha,'teacher_identity':self.gate['teacher_identity'],
            'candidate_scope':'Actual full-chunk BM25/RRF candidates; no source-page injection; weak source-page ranking labels.'}
        return rank
