import numpy as np
from pydantic import BaseModel,Field
from .artifacts import load_artifacts
from .candidates import retrieve,popularity_order
from .data import Snapshot
from .features import features

class Recommendation(BaseModel):
    rank: int=Field(ge=1)
    item_id: str
    category: str
    score: float
    candidate_score: float
class Response(BaseModel):
    user_id: str
    model_version: str
    as_of: str
    strategy: str
    recommendations: list[Recommendation]

class Recommender:
    def __init__(self,root):
        self.config,self.cf,self.booster,self.items,self.history,self.meta,self.version=load_artifacts(root)
        self.snapshot=Snapshot(self.items,self.history,self.meta['serving_as_of'],self.config.event_weights)
    def recommend(self,user,k=10,method='ranker',category=None,snapshot=None):
        if not 1<=k<=self.config.candidate_count: raise ValueError('K must be between 1 and configured candidate_count')
        snapshot=snapshot or self.snapshot
        if method not in ['ranker','popularity','category','cf']: raise ValueError('Invalid recommendation method')
        eligible=snapshot.eligible(user,self.config)
        if category is not None: eligible=[i for i in eligible if snapshot.item_map[i]['category']==category]
        fallback=user not in self.cf.user_index
        if method in ['popularity','category'] or fallback:
            items,values=popularity_order(snapshot,user,eligible,method=='category' or (fallback and self.config.fallback=='category'))
            items=items[:self.config.candidate_count];scores=np.array([values[i] for i in items])
            candidate_scores=dict(zip(items,map(float,self.cf.scores(user,items))))
            strategy='category_popularity' if (method=='category' or (fallback and self.config.fallback=='category')) and len(snapshot.history(user)) else 'global_popularity'
        else:
            items,candidate_scores,strategy=retrieve(snapshot,self.cf,user,self.config)
            if category is not None: items=[i for i in items if i in set(eligible)]
            X=features(snapshot,self.cf,user,items,self.meta['encoders'],self.config)
            scores=self.booster.predict(X) if method=='ranker' and len(items) else self.cf.scores(user,items)
            if method=='ranker': strategy+=' + lambdamart'
        order=sorted(range(len(items)),key=lambda i:(-float(scores[i]),items[i]))[:k]
        result=[Recommendation(rank=rank+1,item_id=items[i],category=snapshot.item_map[items[i]]['category'],
            score=float(scores[i]),candidate_score=float(candidate_scores.get(items[i],0))) for rank,i in enumerate(order)]
        return Response(user_id=user,model_version=self.version,as_of=snapshot.as_of.isoformat(),strategy=strategy,recommendations=result)
