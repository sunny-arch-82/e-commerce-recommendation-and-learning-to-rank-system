from collections import defaultdict
from pathlib import Path
import math
import numpy as np
import pandas as pd
from .core import save_json
from .data import load_data,Snapshot,relevance,fingerprint
from .candidates import retrieve
from .inference import Recommender
from .metrics import ranking_metrics

def evaluate(artifacts,output,split='test',methods=None):
    recommender=Recommender(artifacts);config=recommender.config
    if split not in ['validation','test']: raise ValueError('split must be validation or test')
    _,items,events=load_data(config.data_dir)
    if fingerprint(config.data_dir)!=recommender.meta['dataset_hash']: raise ValueError('Dataset differs from training')
    start,end=(config.validation_start,config.test_start) if split=='validation' else (config.test_start,config.test_end)
    snapshot=Snapshot(items,events,start,config.event_weights);truths,excluded=relevance(snapshot,events,end,config)
    methods=methods or ['popularity','category','cf','ranker'];all_rows=[];errors=[];recommended={m:set() for m in methods};secondary=defaultdict(list); item_slice_counts=defaultdict(lambda:[0,0])
    pop_values=list(snapshot.popularity.values());popular_cut=float(np.quantile(pop_values,.8)) if pop_values else 0.
    user_counts=[len(g) for g in snapshot.user_events.values()];active_cut=float(np.median(user_counts)) if user_counts else 0.
    excluded_total=sum(excluded.values());skipped=0
    for user,truth in sorted(truths.items()):
        relevant={i for i,r in truth.items() if r>=config.relevance_min}
        if not relevant: skipped+=1;continue
        candidates,_,strategy=retrieve(snapshot,recommender.cf,user,config)
        candidate_recall=len(relevant&set(candidates))/len(relevant)
        warm=user in recommender.cf.user_index;active=len(snapshot.history(user))>=active_cut
        for method in methods:
            response=recommender.recommend(user,max(config.ks),method,snapshot=snapshot)
            ranked=[r.item_id for r in response.recommendations];recommended[method].update(ranked)
            counts=defaultdict(int)
            for item in ranked:counts[snapshot.item_map[item]['category']]+=1
            diversity=len(counts)/len(ranked) if ranked else 0.
            concentration=max(counts.values())/len(ranked) if ranked else 0.
            denom=sum(snapshot.popularity.values())+len(snapshot.item_map)
            novelty=float(np.mean([-math.log2((snapshot.popularity.get(i,0)+1)/denom) for i in ranked])) if ranked else 0.
            secondary[method].append({'category_diversity':diversity,'category_concentration':concentration,'novelty_bits':novelty,
                'popular_item_fraction':float(np.mean([snapshot.popularity.get(i,0)>=popular_cut for i in ranked])) if ranked else 0.})
            for k in config.ks:
                metric=ranking_metrics(ranked,truth,k,config.relevance_min)
                for item in relevant:
                    dimensions={'category':snapshot.item_map[item]['category'],
                        'cold_item':str(item not in recommender.cf.item_index),
                        'popularity':'popular' if snapshot.popularity.get(item,0)>=popular_cut else 'long_tail'}
                    for dimension,value in dimensions.items():
                        counts_for_slice=item_slice_counts[(method,k,dimension,value)]
                        counts_for_slice[0]+=int(item in ranked[:k]);counts_for_slice[1]+=1
                all_rows.append({'user_id':user,'method':method,'k':k,'warm':warm,'active':active,
                    'candidate_recall':candidate_recall,'truth_count':len(relevant),**metric})
            if method=='ranker':
                top=ranked[:min(10,max(config.ks))];missing=sorted(relevant-set(candidates))
                for item in sorted(relevant-set(top)):
                    errors.append({'user_id':user,'item_id':item,'failure':'cold_start_fallback_miss' if not warm else 'candidate_missing' if item in missing else 'ranked_below_k',
                        'warm_user':warm,'cold_item':item not in recommender.cf.item_index,'category':snapshot.item_map[item]['category'],
                        'long_tail':snapshot.popularity.get(item,0)<popular_cut,'candidate_strategy':strategy})
    if not all_rows: raise ValueError('No eligible evaluation queries; inspect filters/temporal labels')
    frame=pd.DataFrame(all_rows);metrics=['precision','recall','ndcg','map','hit_rate','mrr']
    summary=[]
    for (method,k),g in frame.groupby(['method','k'],sort=True):
        summary.append({'method':method,'k':int(k),'queries':len(g),**{m:float(g[m].mean()) for m in metrics}})
    slices={}
    for dimension in ['warm','active']:
        slices[dimension]=[]
        for (method,k,value),g in frame.groupby(['method','k',dimension],sort=True):
            slices[dimension].append({'method':method,'k':int(k),'value':bool(value),'queries':len(g),**{m:float(g[m].mean()) for m in metrics}})
    breakdown={'category':{},'cold_item':{},'long_tail':{}}
    for dimension in breakdown:
        values=sorted(set(str(row[dimension]) for row in errors))
        breakdown[dimension]={v:sum(str(row[dimension])==v for row in errors) for v in values}
    report={'split':split,'model_version':recommender.version,'dataset_hash':recommender.meta['dataset_hash'],
        'window':[start,end],'metrics':summary,'slices':slices,'configured_candidate_recall':float(frame[frame.method==methods[0]].candidate_recall.mean()),
        'secondary':{m:{'catalog_coverage':len(recommended[m])/len(snapshot.item_map),
                         **{key:float(np.mean([r[key] for r in secondary[m]])) for key in secondary[m][0]}} for m in methods},
        'excluded_relevant_unavailable_or_already_filtered':excluded_total,'queries_without_eligible_positive_skipped':skipped,
        'error_counts_by_item_slice':breakdown,
        'item_slice_recall':[{'method':method,'k':int(k),'dimension':dimension,'value':value,
            'hits':hits,'targets':total,'micro_recall':hits/total}
            for (method,k,dimension,value),(hits,total) in sorted(item_slice_counts.items())],
        'definitions':{'precision':'hits/K even when fewer than K returned','ndcg':'graded gains 2^relevance-1',
            'map':'binary AP normalized by min(eligible relevant count,K)','mrr':'first relevant reciprocal rank truncated at K',
            'candidate_recall':'configured blended pool; a retrieval ceiling only for warm-user ranker path',
            'target':'observed future implicit events, not randomized relevance or counterfactual utility'}}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    save_json(out/'metrics.json',report);frame.to_csv(out/'per_query.csv',index=False)
    pd.DataFrame(errors,columns=['user_id','item_id','failure','warm_user','cold_item','category','long_tail','candidate_strategy']).to_csv(out/'errors.csv',index=False)
    poor=frame[(frame.method=='ranker')&(frame.k==min(config.ks,key=lambda k:abs(k-10)))].sort_values(['recall','ndcg','user_id'])
    poor.to_csv(out/'poor_users.csv',index=False)
    return {'report':str(out/'metrics.json'),'queries':int(frame.user_id.nunique()),'metrics':summary}
