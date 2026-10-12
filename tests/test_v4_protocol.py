"""Source closure, routing leakage boundaries, and opt-in API contracts."""
import json
import numpy as np
import pytest
from chipclaw.cascade_v4 import features,fit_gain,expected_gain,language
from chipclaw.packets_v4 import closed_table_spans,build_dependency_spans
from chipclaw.layout_codec import encode_layout
from chipclaw.api import Question
def page(text,geometry):
    return {'doc_id':'test','page':1,'text':encode_layout(text),'source_sha256':'a'*64,'url':'https://example.test',
            'title':'Synthetic source','extraction':'pdfplumber-physical-cells-v1','geometry_records':geometry}
def test_physical_closure_keeps_whole_table_and_nearby_notes_with_exact_offsets():
    p=page('1 Electrical characteristics\nGlobal conditions 3 V to 5 V\nParameter    TYP    MAX    Unit\nSupply current    0.3    1    mA\n(1) Typical at 25 C\n2 Next section',[
        {'r':2,'kind':'table','table_index':0},{'r':3,'kind':'table','table_index':0}])
    span=closed_table_spans(p)[0]
    assert span['text']==p['text'][span['start']:span['end']]
    assert {2,3}<={r['r'] for r in span['source_rows']}
    assert '(1)' in span['text'] and 'Global conditions' in span['text'] and 'Next section' not in span['text']
def test_dependency_budget_does_not_slice_an_oversized_table_to_isolated_numbers():
    text='Parameter    TYP    MAX    Unit\n'+''.join(f'Supply current state{i}    0.3    1    mA\n' for i in range(40))
    p=page(text,[{'r':i,'kind':'table','table_index':0} for i in range(41)])
    spans,report=build_dependency_spans([p],'supply current?',256,1,{})
    assert not spans and report['used_chars']==0 and report['rejected_packets']
def test_nested_physical_tables_are_merged_before_contiguous_source_closure():
    p=page('Caption\nOuter header\nInner header\nInner value\nOuter footer',[
        {'r':1,'kind':'table','table_index':0},{'r':4,'kind':'table','table_index':0},
        {'r':2,'kind':'table','table_index':1},{'r':3,'kind':'table','table_index':1}])
    spans=closed_table_spans(p)
    assert len(spans)==1 and spans[0]['closure']['table_indices']==[0,1]
    assert spans[0]['closure']['physical_rows']==[1,2,3,4]
def test_gate_features_do_not_accept_teacher_outputs_or_labels_and_ignore_candidate_ties():
    f=features('What supply current?',np.array([1.,1.,.5]),[5,5,12])
    assert f.shape==(6,) and np.isfinite(f).all() and f[0]==0
    assert features('Question',[0.],[5])[1]==pytest.approx(0,abs=1e-9)
    with pytest.raises(ValueError):features('Q',[float('nan')],[5])
    with pytest.raises(ValueError):features('Q',[1,2],[5])
def test_constant_training_feature_is_bounded_and_gain_model_can_learn_a_negative_signal():
    x=[[0.]*6,[0.]*6,[0.]*6];model=fit_gain(x,[-.1,-.2,-.3])
    assert min(model['scale'])>0 and expected_gain(model,[0.]*6)==pytest.approx(-.2)
def test_language_is_question_derived():
    assert language('典型供电电流是多少?')=='zh'
    assert language('What current?')=='en'
@pytest.mark.parametrize('method',['multilingual_v4','adaptive_v4','domain_teacher_v4','visual_fusion_v4'])
def test_new_methods_are_opt_in_and_require_original_corpus_and_supported_answer_path(method):
    assert Question(question='What current?',doc_id='max3232',corpus='v3',mode='G',retrieval_method=method).retrieval_method==method
    with pytest.raises(ValueError):Question(question='What current?',doc_id='max3232',corpus='v2',retrieval_method=method)
    with pytest.raises(ValueError):Question(question='What current?',doc_id='max3232',corpus='v3',mode='F',retrieval_method=method)
def test_api_g_h_share_one_backend_retriever_without_duplicate_gpu_models(monkeypatch):
    import threading
    from chipclaw import api,retrieval_v4,geometry_v3
    monkeypatch.setattr(geometry_v3, "V3GeometryLoader", lambda: object())
    instances=[];answerers=[]
    class FakeRetriever:
        def __init__(self):instances.append(self)
    class FakeAnswerer:
        def __init__(self,r,source_loader=None,packet_builder=None):self.r=r;answerers.append(self)
        def answer(self,question,doc_id,retrieval_method):return {'metrics':{},'evidence':[],'answer':'contract','refused':False}
    monkeypatch.setattr(api,'v3_answerers',{});monkeypatch.setattr(api,'v3_retrievers',{})
    monkeypatch.setattr(api,'lock',threading.Lock());monkeypatch.setattr(retrieval_v4,'V4Retriever',FakeRetriever)
    monkeypatch.setattr(api,'ReliableAnswerer',FakeAnswerer);monkeypatch.setattr(api,'StructuredEvidenceAnswerer',FakeAnswerer)
    for mode in ['G','H','G']:
        api.ask(Question(question='What current?',doc_id='max3232',corpus='v3',mode=mode,retrieval_method='adaptive_v4'))
    assert len(instances)==1 and len(answerers)==2 and answerers[0].r is answerers[1].r
    assert not api.lock.locked()
def test_evidence_api_rejects_unbounded_claims_before_model_loading():
    from chipclaw.api import EvidenceQuestion
    with pytest.raises(ValueError):EvidenceQuestion(question='What current?',doc_id='max3232',claims=['x'*401])
    with pytest.raises(ValueError):EvidenceQuestion(question='What current?',doc_id='max3232',budget_chars=25000)
    with pytest.raises(ValueError):EvidenceQuestion(question='What current?',doc_id='max3232',claims=['x']*9)
def test_domain_ranking_uses_raw_logit_order_when_sigmoid_probabilities_are_saturated(monkeypatch):
    from chipclaw.domain_teacher_v4 import DomainTeacher
    from chipclaw.teacher_v3 import QwenTeacher
    records=[{'score':1.,'logit_margin':21.},{'score':1.,'logit_margin':25.}]
    monkeypatch.setattr(QwenTeacher,'predict',lambda self,pairs,batch_size=8,return_records=False:records)
    model=DomainTeacher.__new__(DomainTeacher)
    assert model.predict([('q','a'),('q','b')])==[21.,25.]
    assert model.predict([('q','a'),('q','b')],return_records=True)==records

def test_evidence_only_startup_does_not_load_legacy_dense_models(monkeypatch):
    from fastapi.testclient import TestClient
    from chipclaw import api
    calls=[]
    monkeypatch.setenv('CHIPCLAW_API_WARMUP','none')
    monkeypatch.setattr(api,'retriever',None)
    monkeypatch.setattr(api,'Retriever',lambda:calls.append('legacy'))
    with TestClient(api.app) as client:
        assert client.get('/health').status_code==200
    assert not calls and api.retriever is None

def test_legacy_backend_remains_available_after_lazy_startup(monkeypatch):
    from chipclaw import api
    backend=object()
    monkeypatch.setattr(api,'Retriever',lambda:backend)
    monkeypatch.setattr(api,'EvidencePlanner',lambda r:('planner',r))
    monkeypatch.setattr(api,'retriever',None)
    monkeypatch.setattr(api,'planners',{})
    api.initialize_legacy_retriever()
    assert api.retriever is backend and api.planners['main']==('planner',backend)

def test_invalid_startup_configuration_fails_before_model_loading(monkeypatch):
    from fastapi.testclient import TestClient
    from chipclaw import api
    monkeypatch.setenv('CHIPCLAW_API_WARMUP','invalid')
    with pytest.raises(ValueError,match='CHIPCLAW_API_WARMUP'):
        with TestClient(api.app):pass

def test_same_retriever_can_switch_between_legacy_and_v4_ranking_without_protocol_conflict():
    from chipclaw.retrieval_v4 import V4Retriever
    calls=[]
    class Legacy:
        last_metrics={}
        def rank(self,question,ids,texts,method):calls.append(method);return list(range(len(ids)))
    class Current:
        last_metrics={}
        def rank(self,question,ids,texts,pages,method):calls.append(method);return list(range(len(ids)))
    r=V4Retriever.__new__(V4Retriever)
    r.ranker=Legacy();r.v4_ranker=Current();r.visual=None;r.v3_events=[];r.retrieval_events=[]
    r.chunks=[{'id':'a','doc_id':'max3232','page':5,'text':'Supply current','source_sha256':'a'*64}]
    r.lexical_candidates=lambda question,doc_id:[0]
    for method in ['student_v3','adaptive_v4','student_v3']:
        assert r.retrieve('What current?','max3232',method)[0]['id']=='a'
    assert calls==['student_v3','adaptive_v4','student_v3']

def test_chinese_no_load_condition_is_retained_in_english_source_selection():
    from chipclaw.packets_v4 import selection_terms
    en,_=selection_terms('Typical supply current with no load?')
    zh,audit=selection_terms('典型供电电流在空载时是多少？')
    assert {'supply','current','no','load'}<=en&zh
    assert audit==[{'source':'空载','canonical':'no load'}]
    # No numeric answer is introduced by this requirement expansion.
    assert not any(t.replace('.','').isdigit() for t in zh)
