from pathlib import Path
import numpy as np
import pandas as pd
from .core import EVENTS,utc,sha

def load_data(root):
    root=Path(root);users=pd.read_csv(root/'users.csv',dtype={'user_id':str})
    items=pd.read_csv(root/'items.csv',dtype={'item_id':str})
    events=pd.read_csv(root/'interactions.csv',dtype={'user_id':str,'item_id':str,'interaction_id':str})
    required=[(users,{'user_id','created_at'}),(items,{'item_id','category','brand','price','created_at','available'}),
              (events,{'interaction_id','user_id','item_id','timestamp','event_type'})]
    for frame,columns in required:
        if not columns.issubset(frame.columns) or frame[list(columns)].isna().any().any(): raise ValueError('Missing columns or values')
    for frame,key in [(users,'user_id'),(items,'item_id'),(events,'interaction_id')]:
        if frame[key].duplicated().any(): raise ValueError(f'Duplicate {key}')
    if not len(users) or not len(items) or not len(events): raise ValueError('Empty data table')
    if not np.isfinite(items.price).all() or (items.price<=0).any(): raise ValueError('Invalid item prices')
    for frame,col in [(users,'created_at'),(items,'created_at'),(events,'timestamp')]:
        frame[col]=frame[col].map(utc)
    if not set(events.user_id)<=set(users.user_id) or not set(events.item_id)<=set(items.item_id): raise ValueError('Unknown event entity')
    if not set(events.event_type)<=set(EVENTS): raise ValueError('Unknown event type')
    availability=items.available.astype(str).str.lower()
    if not availability.isin(['true','false','1','0']).all(): raise ValueError('Invalid available flag')
    items['available']=availability.isin(['true','1'])
    joined=events.merge(items[['item_id','created_at']],on='item_id').merge(users[['user_id','created_at']],on='user_id',suffixes=('_item','_user'))
    if ((joined.timestamp<joined.created_at_item)|(joined.timestamp<joined.created_at_user)).any(): raise ValueError('Interaction predates entity creation')
    events=events.sort_values(['timestamp','interaction_id'],kind='stable').reset_index(drop=True)
    # Generator preference/price sensitivity/activity are intentionally inaccessible to feature code.
    return users[['user_id','created_at']],items,events

def fingerprint(root):
    from hashlib import sha256
    return sha256(''.join(sha(Path(root)/name) for name in ['users.csv','items.csv','interactions.csv']).encode()).hexdigest()

def audit(root,config=None):
    users,items,events=load_data(root)
    report={'passed':True,'users':len(users),'items':len(items),'interactions':len(events),'dataset_hash':fingerprint(root),
            'event_distribution':events.event_type.value_counts().to_dict(),'range':[events.timestamp.min().isoformat(),events.timestamp.max().isoformat()],
            'limitations':'Exact event IDs validated; duplicate exposures and declared catalog availability require source review.'}
    if config:
        bounds=[utc(config.candidate_cutoff),utc(config.validation_start),utc(config.test_start),utc(config.test_end)]
        report['period_counts']={'candidate_history':int((events.timestamp<bounds[0]).sum()),
            'ranker_labels':int(((events.timestamp>=bounds[0])&(events.timestamp<bounds[1])).sum()),
            'validation':int(((events.timestamp>=bounds[1])&(events.timestamp<bounds[2])).sum()),
            'test':int(((events.timestamp>=bounds[2])&(events.timestamp<bounds[3])).sum())}
        if min(report['period_counts'].values())<1: raise ValueError('Every temporal period needs events')
    return report

class Snapshot:
    def __init__(self,items,events,as_of,event_weights):
        self.as_of=utc(as_of)
        self.items=items[(items.created_at<self.as_of)&items.available].copy().sort_values('item_id')
        self.item_map=self.items.set_index('item_id').to_dict('index')
        self.events=events[events.timestamp<self.as_of].copy()
        self.events['weight']=self.events.event_type.map(event_weights).astype(float)
        self.user_events={user:g.copy() for user,g in self.events.groupby('user_id',sort=True)}
        self.empty=self.events.iloc[:0]
        self.popularity=self.events.groupby('item_id').weight.sum().to_dict()
        self.item_counts=self.events.groupby('item_id').size().to_dict()
        self.item_purchases=self.events[self.events.event_type=='purchase'].groupby('item_id').size().to_dict()
        self.max_timestamp=self.events.timestamp.max() if len(self.events) else None
    def history(self,user): return self.user_events.get(user,self.empty)
    def eligible(self,user,config):
        history=self.history(user)
        excluded=set(history.item_id) if config.exclude_seen else set(history.loc[history.event_type=='purchase','item_id']) if config.exclude_purchased else set()
        return sorted(set(self.item_map)-excluded)
    def affinity(self,user):
        history=self.history(user)
        if not len(history): return {}
        cats=history.item_id.map(lambda i:self.item_map.get(i,{}).get('category','unknown'))
        totals=history.groupby(cats).weight.sum();denominator=float(totals.sum())
        return {str(k):float(v/denominator) for k,v in totals.items()}

def relevance(snapshot,events,end,config):
    end=utc(end)
    future=events[(events.timestamp>=snapshot.as_of)&(events.timestamp<end)].copy()
    future['relevance']=future.event_type.map(EVENTS)
    raw={u:g.groupby('item_id').relevance.max().to_dict() for u,g in future.groupby('user_id',sort=True)}
    eligible={};excluded={}
    for user,truth in raw.items():
        available=set(snapshot.eligible(user,config));eligible[user]={i:int(r) for i,r in truth.items() if i in available}
        excluded[user]=len([i for i,r in truth.items() if r>=config.relevance_min and i not in available])
    return eligible,excluded
