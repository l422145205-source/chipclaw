"""Local MiniLM-first cascade. Gate identity is tied to dev-only calibration."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from .config import ROOT
from .teacher_v3 import REVISION

def ordered(scores,ids):return sorted(range(len(scores)),key=lambda i:(-float(scores[i]),ids[i]))

def cascade_order(student_scores,ids,teacher_scores=None,top_n=6):
    rank=ordered(student_scores,ids)
    if teacher_scores is None:return rank
    head=rank[:top_n]
    if len(teacher_scores)!=len(head):raise ValueError('Teacher shortlist cardinality mismatch')
    # Sort the shortlisted head alone; incomparable raw scales are never mixed.
    return [head[i] for i in ordered(teacher_scores,[ids[j] for j in head])]+rank[top_n:]

class V3Student:
    def __init__(self,path=None,device='cuda:0'):
        import torch
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        if path is None:
            selected=json.loads((ROOT/'reports/minilm-v3/selected_model.json').read_text())
            if selected['selection_split']!='dev':raise ValueError('Student arm must be selected on dev')
            path=ROOT/'.cache/v3/student'/selected['arm']/'best'
        self.path=Path(path);self.device=device
        manifest=json.loads((self.path/'training_manifest.json').read_text(encoding='utf-8'))
        self.weight_sha=hashlib.sha256((self.path/'model.safetensors').read_bytes()).hexdigest()
        if self.weight_sha!=manifest['weight_sha256']:raise ValueError('Student weight fingerprint mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(self.path,local_files_only=True).to(device).eval()
        self.last_metrics=None

    def predict(self,pairs,batch_size=16):
        import torch
        start=time.perf_counter();scores=[]
        with torch.inference_mode():
            for i in range(0,len(pairs),batch_size):
                part=pairs[i:i+batch_size]
                features=self.tokenizer([p[0] for p in part],[p[1] for p in part],padding=True,truncation='only_second',max_length=512,return_tensors='pt').to(self.device)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):
                    scores+=self.model(**features).logits[:,0].float().cpu().tolist()
        if self.device.startswith('cuda'):torch.cuda.synchronize()
        self.last_metrics={'pairs':len(pairs),'forward_batches':(len(pairs)+batch_size-1)//batch_size,'latency_ms':(time.perf_counter()-start)*1000}
        return scores

class CascadeRanker:
    def __init__(self,student=None,teacher=None,gate_path=None):
        self.student=student or V3Student();self.teacher=teacher
        self.gate=json.loads(Path(gate_path or ROOT/'reports/cascade-v3/gate.json').read_text())
        if self.gate['student_weight_sha256']!=self.student.weight_sha or self.gate['teacher_revision']!=REVISION:
            raise ValueError('Gate/model identity mismatch')
        if self.gate['selection_split']!='dev':raise ValueError('Gate must use dev only')
        if not np.isfinite(self.gate['gap_threshold']) or not 1<=self.gate['teacher_top_n']<=16:
            raise ValueError('Invalid bounded dev gate')
        self.last_metrics=None

    def rank(self,question,ids,texts,mode='cascade_v3'):
        if len(ids)!=len(texts) or not ids:raise ValueError('Nonempty aligned candidate lists required')
        start=time.perf_counter();teacher_pairs=0;teacher_batches=0;student_batches=0
        if mode=='teacher_v3':
            if self.teacher is None:
                from .teacher_v3 import QwenTeacher
                self.teacher=QwenTeacher()
            scores=self.teacher.predict([(question,p) for p in texts]);rank=ordered(scores,ids)
            teacher_pairs=len(ids);teacher_batches=self.teacher.last_metrics['forward_batches'];gap=None
        else:
            scores=self.student.predict([(question,p) for p in texts]);student_batches=self.student.last_metrics['forward_batches']
            rank=ordered(scores,ids);gap=float(scores[rank[0]]-scores[rank[1]]) if len(rank)>1 else float('inf')
            if mode not in {'student_v3','cascade_v3'}:raise ValueError('Unknown local ranker mode')
            if mode=='cascade_v3' and gap<=self.gate['gap_threshold']:
                if self.teacher is None:
                    from .teacher_v3 import QwenTeacher
                    self.teacher=QwenTeacher()
                head=rank[:self.gate['teacher_top_n']]
                teacher_scores=self.teacher.predict([(question,texts[i]) for i in head])
                rank=cascade_order(scores,ids,teacher_scores,self.gate['teacher_top_n'])
                teacher_pairs=len(head);teacher_batches=self.teacher.last_metrics['forward_batches']
        self.last_metrics={'latency_ms':(time.perf_counter()-start)*1000,'student_forward_batches':student_batches,
                           'teacher_forward_batches':teacher_batches,'teacher_pairs':teacher_pairs,'candidate_pairs':len(ids),
                           'gap':gap,'gate_split':'dev','mode':mode}
        return rank
