import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from .core import config_from

def main():
    p=argparse.ArgumentParser(prog='rankforge');sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('generate');q.add_argument('--output',default='data/generated');q.add_argument('--users',type=int,default=60);q.add_argument('--items',type=int,default=100);q.add_argument('--events',type=int,default=4000);q.add_argument('--seed',type=int,default=42);q.add_argument('--popularity-bias',type=float,default=1.)
    q=sub.add_parser('audit');q.add_argument('--config',default='configs/demo.yaml')
    q=sub.add_parser('train');q.add_argument('--config',default='configs/demo.yaml')
    q=sub.add_parser('evaluate');q.add_argument('--artifacts',default='artifacts/demo');q.add_argument('--output',default='outputs/test');q.add_argument('--split',choices=['validation','test'],default='test')
    q=sub.add_parser('recommend');q.add_argument('--artifacts',default='artifacts/demo');q.add_argument('--user',required=True);q.add_argument('--k',type=int,default=10);q.add_argument('--method',choices=['ranker','popularity','category','cf'],default='ranker');q.add_argument('--category')
    q=sub.add_parser('batch');q.add_argument('--artifacts',default='artifacts/demo');q.add_argument('--users',default='data/sample/users.csv');q.add_argument('--output',default='outputs/recommendations.csv');q.add_argument('--k',type=int,default=10)
    q=sub.add_parser('tune');q.add_argument('--config',default='configs/demo.yaml');q.add_argument('--space',default='configs/search.yaml');q.add_argument('--trials',type=int,default=8);q.add_argument('--output',default='outputs/tuning')
    q=sub.add_parser('compare');q.add_argument('--configs',nargs='+',required=True);q.add_argument('--output',default='outputs/ablations')
    a=p.parse_args()
    try:
        if a.command=='generate':
            from .generator import generate
            result=generate(a.output,a.users,a.items,a.events,a.seed,a.popularity_bias)
        elif a.command=='audit':
            from .data import audit
            config=config_from(a.config);result=audit(config.data_dir,config)
        elif a.command=='train':
            from .training import train
            result=train(config_from(a.config))
        elif a.command=='evaluate':
            from .evaluation import evaluate
            result=evaluate(a.artifacts,a.output,a.split)
        elif a.command=='recommend':
            from .inference import Recommender
            result=Recommender(a.artifacts).recommend(a.user,a.k,a.method,a.category).model_dump()
        elif a.command=='batch':
            from .inference import Recommender
            model=Recommender(a.artifacts);users=pd.read_csv(a.users,dtype={'user_id':str})
            if 'user_id' not in users or users.user_id.isna().any():raise ValueError('users CSV requires user_id')
            rows=[]
            for user in users.user_id.drop_duplicates():
                response=model.recommend(user,a.k)
                rows.extend({'user_id':user,**r.model_dump(),'model_version':response.model_version,'strategy':response.strategy,'as_of':response.as_of} for r in response.recommendations)
            out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);frame=pd.DataFrame(rows)
            if out.suffix=='.parquet':frame.to_parquet(out,index=False)
            else:frame.to_csv(out,index=False)
            result={'output':str(out),'recommendations':len(rows),'users':int(users.user_id.nunique())}
        elif a.command=='tune':
            from .experiments import tune
            result=tune(config_from(a.config),yaml.safe_load(Path(a.space).read_text()),a.trials,a.output)
        else:
            from .experiments import compare
            result=compare(a.configs,a.output)
        print(json.dumps(result,indent=2))
    except (ValueError,OSError) as exc:p.exit(2,f'error: {exc}\n')
