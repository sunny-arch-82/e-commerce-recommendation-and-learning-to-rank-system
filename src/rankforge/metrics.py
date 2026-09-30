import math

def ranking_metrics(recommendations,relevance,k,minimum=2):
    if k<1: raise ValueError('K must be positive')
    if len(set(recommendations))!=len(recommendations): raise ValueError('Recommendations must be unique')
    relevant={i for i,r in relevance.items() if r>=minimum}
    if not relevant: return None
    ranked=recommendations[:k];hits=[int(i in relevant) for i in ranked]
    precision=sum(hits)/k;recall=sum(hits)/len(relevant)
    ap=sum(sum(hits[:n+1])/(n+1) for n,hit in enumerate(hits) if hit)/min(len(relevant),k)
    dcg=sum((2**relevance.get(i,0)-1)/math.log2(n+2) for n,i in enumerate(ranked))
    ideal=sorted(relevance.values(),reverse=True)[:k]
    idcg=sum((2**r-1)/math.log2(n+2) for n,r in enumerate(ideal))
    reciprocal=next((1/(n+1) for n,hit in enumerate(hits) if hit),0.)
    return {'precision':precision,'recall':recall,'ndcg':dcg/idcg if idcg else 0.,'map':ap,
            'hit_rate':float(any(hits)),'mrr':reciprocal}
