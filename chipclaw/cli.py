import argparse
import json
import time
from .config import SOURCES, HOLDOUT_SOURCES, HOLDOUT_INDEX, EXPANSION_SOURCES, EXPANDED_INDEX, DOCUMENT_SOURCES


def main():
    from .corpus_v3 import SOURCES_V3
    from .retrieval_v3 import METHODS_V3
    from .retrieval_v4 import METHODS_V4
    parser=argparse.ArgumentParser(description='ChipClaw local RAG')
    sub=parser.add_subparsers(dest='command',required=True)
    ingest=sub.add_parser('ingest');ingest.add_argument('--download',action='store_true')
    sub.add_parser('index')
    evaluate=sub.add_parser('evaluate')
    evaluate.add_argument('--split',choices=['dev','test'],default='test')
    evaluate.add_argument('--name')
    evaluate.add_argument('--modes',nargs='+',choices=['A','B','C','D'],default=['A','B','C','D'])
    ask=sub.add_parser('ask');ask.add_argument('question');ask.add_argument('--doc',choices=list(SOURCES_V3),required=True)
    ask.add_argument('--corpus',choices=['v2','v3'],default='v2')
    ask.add_argument('--mode',choices=['A','B','C','D','E','F','G','H'],default='G')
    ask.add_argument('--selector',choices=['baseline','compiled','protected','protected_gate'],default='protected',help='F research path selector')
    ask.add_argument('--retrieval-method',choices=['rewrite_hybrid','domain_rerank','protected_hybrid',*sorted(METHODS_V3|METHODS_V4)],default='rewrite_hybrid',help='G/H retrieval; v3/v4 methods require --corpus v3')
    args=parser.parse_args()
    if args.command=='ask':
        if args.corpus=='v2' and args.doc not in DOCUMENT_SOURCES:parser.error('Document requires --corpus v3')
        if args.corpus=='v3' and args.mode not in {'G','H'}:parser.error('v3 corpus is implemented for G/H only')
        if args.retrieval_method in METHODS_V3|METHODS_V4 and args.corpus!='v3':parser.error('v3/v4 retrieval requires --corpus v3')
    if args.command == 'ask' and args.retrieval_method != 'rewrite_hybrid' and args.mode not in {'G', 'H'}:
        parser.error('domain_rerank and protected_hybrid retrieval are implemented for G/H only')
    if args.command=='ingest':
        from .ingest import build
        result=build(args.download)
    elif args.command=='index':
        from .retrieval import Retriever
        r=Retriever();result={'chunks':len(r.chunks),'dimension':r.vectors.shape[1]}
    elif args.command=='evaluate':
        from .evaluate import run
        result=run(args.split,args.name,tuple(args.modes))
    else:
        from .retrieval import Retriever
        from .generation import generate,verify
        request_start=time.perf_counter()
        if args.corpus=='v3':
            from .retrieval_v3 import V3Retriever
            from .retrieval_v4 import V4Retriever
            r=(V4Retriever if args.retrieval_method in METHODS_V4 else V3Retriever)()
        else:r=Retriever(index_dir=EXPANDED_INDEX if args.mode in {'G','H'} or args.doc in EXPANSION_SOURCES else HOLDOUT_INDEX if args.doc in HOLDOUT_SOURCES else None)
        if args.mode in {'G','H'}:
            from .reliable import ReliableAnswerer
            from .orchestration import StructuredEvidenceAnswerer
            init_ms=(time.perf_counter()-request_start)*1000
            kwargs={}
            if args.corpus=='v3':
                from .reliable import SourceLoader
                from .corpus_v3 import V3_RAW
                from .geometry_v3 import V3GeometryLoader
                kwargs['source_loader']=SourceLoader(V3_RAW,sources=SOURCES_V3,geometry_loader=V3GeometryLoader())
                if args.retrieval_method in METHODS_V4:
                    from .packets_v4 import build_dependency_spans
                    kwargs['packet_builder']=build_dependency_spans
            result=(StructuredEvidenceAnswerer(r,**kwargs) if args.mode=='H' else ReliableAnswerer(r,**kwargs)).answer(args.question,args.doc,retrieval_method=args.retrieval_method)
            result['metrics']['retriever_init_ms']=init_ms
            result['metrics']['request_ms']=(time.perf_counter()-request_start)*1000
            hits=result['evidence']
        elif args.mode=='F':
            from .planning import EvidencePlanner,answer_requirements
            from .budget_gated import BudgetAwarePlanner
            hits,trace=BudgetAwarePlanner(r).retrieve(args.question,args.doc,method=args.selector)
            result=answer_requirements(args.question,hits,trace)
        else:
            hits=r.context_windows(r.retrieve(args.question,args.doc))
            result=generate(args.question,hits,'C' if args.mode in {'D','E'} else args.mode)
            if args.mode=='D':result=verify(result,hits)
            if args.mode=='E':
                from .repair import verify_with_repair
                result=verify_with_repair(result,hits)
        result['evidence']=hits
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
