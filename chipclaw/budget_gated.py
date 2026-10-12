"""Optional operational shortcut; the frozen v3 planner remains unchanged."""
import copy
from .planning import EvidencePlanner

class BudgetAwarePlanner(EvidencePlanner):
    def retrieve(self, question, doc_id, method='protected', saved_plan=None, english_query=None):
        if method!='protected_gate':
            return super().retrieve(question,doc_id,method,saved_plan,english_query)
        baseline,first=super().retrieve(question,doc_id,'baseline',saved_plan,english_query)
        chars=sum(len(e['text']) for e in baseline)
        impossible=len(baseline)>=self.max_pages or self.budget_chars-chars<96
        if impossible:
            trace=copy.deepcopy(first)
            trace['selection'].update({'effective_selector':'protected_gate',
                'mandatory_baseline':[e['doc_id']+':'+str(e['page']) for e in baseline],
                'baseline_chars':chars,'extra_pages':0,'objective':None,
                'guarantee':'Baseline windows unchanged; no room for the protected selector to add a new page.'})
            trace['budget_gate']={'skipped_compiler':True,'reason':'No page slot or fewer than 96 remaining characters',
                'note':'No generated-answer guarantee. This shortcut precedes support compilation.'}
            return baseline,trace
        plan=saved_plan if saved_plan is not None else self.plans[(question,doc_id)]
        evidence,trace=super().retrieve(question,doc_id,'protected',plan,first['english_query'])
        trace['cache']['plan']=first['cache']['plan'];trace['cache']['rewrite']=first['cache']['rewrite']
        trace['latency_ms']+=first['latency_ms']
        trace['selection']['effective_selector']='protected_gate'
        trace['budget_gate']={'skipped_compiler':False,'reason':'Baseline leaves room for supplementary evidence',
            'note':'Feasible branch repeats CPU retrieval with cached embeddings; measure this overhead separately.'}
        return evidence,trace
