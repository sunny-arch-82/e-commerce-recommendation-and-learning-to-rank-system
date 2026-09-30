from hashlib import sha256
import numpy as np
import pandas as pd
from .candidates import retrieve,sample_negatives
from .features import features,names
from .data import relevance

def query_seed(seed,user): return seed+int(sha256(user.encode()).hexdigest()[:8],16)

def build_queries(snapshot,events,end,cf,codes,config,training=False):
    truths,_=relevance(snapshot,events,end,config);frames=[];labels=[];groups=[];metadata=[]
    for user,truth in sorted(truths.items()):
        if not truth: continue
        candidates,_,_=retrieve(snapshot,cf,user,config)
        if training:
            # Label injection belongs ONLY to training. Evaluation/validation never inject positives.
            positives=sorted(truth)
            negatives=sample_negatives(snapshot.eligible(user,config),positives,candidates,snapshot.popularity,
                config.negatives_per_query,config.negative_strategy,query_seed(config.seed,user))
            items=positives+negatives
        else: items=candidates
        target=[truth.get(i,0) for i in items]
        if len(items)<2 or len(set(target))<2: continue
        frame=features(snapshot,cf,user,items,codes,config)
        frames.append(frame);labels.extend(target);groups.append(len(items));metadata.extend({'user_id':user,'item_id':i} for i in items)
    if not frames: raise ValueError('No informative ranking groups; generate more data or review temporal windows')
    X=pd.concat(frames,ignore_index=True);y=np.array(labels,dtype=int)
    if sum(groups)!=len(X): raise ValueError('Ranking group sizes do not sum to rows')
    return X,y,groups,pd.DataFrame(metadata)

def fit_ranker(X,y,groups,V,vy,vg,config):
    import lightgbm as lgb
    ranker=lgb.LGBMRanker(objective='lambdarank',n_estimators=config.ranker_trees,
        learning_rate=config.ranker_learning_rate,num_leaves=config.ranker_leaves,
        min_child_samples=config.ranker_min_child_samples,reg_lambda=config.ranker_reg_lambda,
        random_state=config.seed,n_jobs=config.ranker_threads,verbosity=-1,deterministic=True,force_col_wise=True,
        label_gain=[0,1,3,7,15])
    ranker.fit(X,y,group=groups,eval_set=[(V,vy)],eval_group=[vg],eval_at=[min(10,config.candidate_count)],
        feature_name=names(config),callbacks=[lgb.early_stopping(10,verbose=False)])
    return ranker.booster_
