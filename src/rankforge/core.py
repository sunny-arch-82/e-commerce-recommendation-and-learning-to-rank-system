import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml
from pydantic import BaseModel,ConfigDict,Field,model_validator

EVENTS={'view':1,'click':2,'add_to_cart':3,'purchase':4}
class Config(BaseModel):
    model_config=ConfigDict(extra='forbid')
    data_dir: str='data/sample'
    output_dir: str='artifacts/demo'
    seed: int=42
    candidate_cutoff: str='2026-01-21T00:00:00Z'
    validation_start: str='2026-01-31T00:00:00Z'
    test_start: str='2026-02-10T00:00:00Z'
    test_end: str='2026-02-20T00:00:00Z'
    event_weights: dict[str,float]=Field(default_factory=lambda:{'view':1.,'click':2.,'add_to_cart':4.,'purchase':8.})
    factors: int=Field(16,ge=2,le=256)
    bpr_epochs: int=Field(5,ge=1)
    bpr_learning_rate: float=Field(.03,gt=0)
    bpr_regularization: float=Field(.01,ge=0)
    candidate_count: int=Field(50,ge=5,le=1000)
    candidate_source: str='collaborative'
    fallback: str='category'
    exclude_purchased: bool=True
    exclude_seen: bool=False
    freshness_fraction: float=Field(.1,ge=0,le=.5)
    popularity_fraction: float=Field(.2,ge=0,le=.5)
    negative_strategy: str='hard'
    negatives_per_query: int=Field(40,ge=1,le=1000)
    feature_set: str='rich'
    include_cf_score: bool=True
    relevance_min: int=Field(2,ge=1,le=4)
    ks: list[int]=Field(default_factory=lambda:[5,10,20])
    ranker_trees: int=Field(80,ge=1)
    ranker_learning_rate: float=Field(.05,gt=0)
    ranker_leaves: int=Field(15,ge=2)
    ranker_min_child_samples: int=Field(10,ge=1)
    ranker_reg_lambda: float=Field(1.,ge=0)
    ranker_threads: int=Field(2,ge=1)
    @model_validator(mode='after')
    def valid(self):
        if set(self.event_weights)!=set(EVENTS) or any(v<=0 for v in self.event_weights.values()):
            raise ValueError('Provide positive weights for all four event types')
        times=[utc(getattr(self,k)) for k in ['candidate_cutoff','validation_start','test_start','test_end']]
        if not all(a<b for a,b in zip(times,times[1:])): raise ValueError('Temporal boundaries must increase strictly')
        if self.candidate_source not in ['collaborative','popularity']: raise ValueError('Invalid candidate_source')
        if self.fallback not in ['category','popularity']: raise ValueError('Invalid fallback')
        if self.negative_strategy not in ['random','popularity','hard']: raise ValueError('Invalid negative_strategy')
        if self.feature_set not in ['basic','rich']: raise ValueError('Invalid feature_set')
        if not self.ks or min(self.ks)<1 or max(self.ks)>self.candidate_count: raise ValueError('K must be positive and <= candidate_count')
        if self.freshness_fraction+self.popularity_fraction>.8: raise ValueError('Reserve at least 20% for primary retrieval')
        return self

def utc(value):
    t=pd.Timestamp(value)
    if t.tzinfo is None: raise ValueError('Timestamps must include a timezone')
    return t.tz_convert('UTC')

def config_from(path): return Config.model_validate(yaml.safe_load(Path(path).read_text()))
def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
