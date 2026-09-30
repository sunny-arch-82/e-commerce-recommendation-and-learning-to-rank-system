import numpy as np

def popularity_order(snapshot,user,eligible,category=False):
    affinity=snapshot.affinity(user) if category else {}
    def score(item):
        category=snapshot.item_map[item]['category']
        return float(snapshot.popularity.get(item,0))*(1+3*affinity.get(category,0))
    return sorted(eligible,key=lambda item:(-score(item),item)),{item:score(item) for item in eligible}

def retrieve(snapshot,cf,user,config,count=None):
    count=count or config.candidate_count;eligible=snapshot.eligible(user,config)
    pop,pop_scores=popularity_order(snapshot,user,eligible,config.fallback=='category')
    scores=dict(zip(eligible,map(float,cf.scores(user,eligible))))
    cold=user not in cf.user_index
    primary=pop if cold or config.candidate_source=='popularity' else sorted(eligible,key=lambda i:(-scores[i],i))
    fresh=sorted(eligible,key=lambda i:(-snapshot.item_map[i]['created_at'].timestamp(),i))
    fresh_n=int(count*config.freshness_fraction);pop_n=int(count*config.popularity_fraction)
    ordered=primary[:max(1,count-fresh_n-pop_n)]+pop[:pop_n]+fresh[:fresh_n]+primary+pop
    result=list(dict.fromkeys(ordered))[:count]
    strategy=('category_popularity' if config.fallback=='category' and len(snapshot.history(user)) else 'global_popularity') if cold else 'configured_popularity' if config.candidate_source=='popularity' else 'weighted_bpr'
    return result,scores,strategy

def sample_negatives(eligible,positive,candidates,popularity,n,strategy,seed):
    available=sorted(set(eligible)-set(positive))
    if not available: return []
    rng=np.random.default_rng(seed);n=min(n,len(available))
    if strategy=='hard':
        hard=[i for i in candidates if i in set(available)]
        result=list(dict.fromkeys(hard))[:n]
        rest=sorted(set(available)-set(result))
        if len(result)<n: result+=rng.choice(rest,size=n-len(result),replace=False).tolist()
        return result
    if strategy not in ['random','popularity']: raise ValueError('Invalid negative strategy')
    weights=np.array([popularity.get(i,0)+1 for i in available],dtype=float) if strategy=='popularity' else None
    if weights is not None: weights/=weights.sum()
    return rng.choice(available,size=n,replace=False,p=weights).tolist()
