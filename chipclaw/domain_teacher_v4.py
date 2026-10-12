"""Domain LoRA yes/no ranking adapter; the public base weights remain unchanged."""
import hashlib,json
from .config import ROOT
from .teacher_v3 import QwenTeacher
INSTRUCTION='Retrieve chip-manual evidence that directly answers the question. Match the parameter, min/typ/max column, unit, operating mode and complete test conditions. Topic overlap or a correct value under a different condition is insufficient.'
class DomainTeacher(QwenTeacher):
    def __init__(self,device='cuda:0',max_length=512,path=None):
        super().__init__(device,max_length,INSTRUCTION)
        from peft import PeftModel
        self.adapter_path=path or ROOT/'.cache/v4/domain_lora/best'
        self.adapter_manifest=json.loads((self.adapter_path/'training_manifest.json').read_text(encoding='utf-8'))
        with (self.adapter_path/'adapter_model.safetensors').open('rb') as f:self.adapter_sha=hashlib.file_digest(f,'sha256').hexdigest()
        if self.adapter_sha!=self.adapter_manifest['adapter_sha256']:raise ValueError('Domain adapter fingerprint mismatch')
        if self.adapter_manifest['base_sha256']!=self.weight_sha:raise ValueError('Adapter/base identity mismatch')
        self.model=PeftModel.from_pretrained(self.model,self.adapter_path,is_trainable=False).eval()
    def predict(self,pairs,batch_size=16,return_records=False):
        records=super().predict(pairs,batch_size,return_records=True)
        # Float32 sigmoid probabilities saturate near one after ranking LoRA.
        # Raw yes-minus-no logits preserve relative order without mixing scales.
        return records if return_records else [r['logit_margin'] for r in records]
    def identity(self):return {**super().identity(),'kind':'domain_lora_v4','adapter_sha256':self.adapter_sha,
        'selection_split':self.adapter_manifest['selection_split'],'rank':self.adapter_manifest['rank'],
        'score_space':'raw_yes_minus_no_logits',
        'interpretation':'Actually trained domain ranking preference adapter; not generative SFT or factual verification.'}
