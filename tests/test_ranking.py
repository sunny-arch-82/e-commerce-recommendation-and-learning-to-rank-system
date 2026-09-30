import numpy as np
from rankforge.data import load_data,Snapshot,relevance
from rankforge.collaborative import WeightedBPR
from rankforge.features import encoders,features,names
from rankforge.ranking import build_queries

def test_features_and_queries(config):
    _,items,events=load_data(config.data_dir);snap=Snapshot(items,events,config.candidate_cutoff,config.event_weights)
    cf=WeightedBPR(4,7).fit(snap.events,epochs=1);codes=encoders(snap.items)
    X,y,groups,meta=build_queries(snap,events,config.validation_start,cf,codes,config,True)
    assert sum(groups)==len(X)==len(y)==len(meta)
    assert list(X.columns)==names(config) and np.isfinite(X.to_numpy()).all()
    start=0
    for size in groups:
        assert meta.iloc[start:start+size].user_id.nunique()==1
        start+=size
    # Future label changes cannot alter historical feature values.
    user=cf.user_ids[0];ids=snap.eligible(user,config)[:3]
    values=features(snap,cf,user,ids,codes,config)
    modified=events.copy();modified.loc[modified.timestamp>=snap.as_of,'event_type']='purchase'
    other=Snapshot(items,modified,config.candidate_cutoff,config.event_weights)
    np.testing.assert_allclose(values,features(other,cf,user,ids,codes,config))

def test_validation_no_positive_injection(config):
    from rankforge.candidates import retrieve
    _,items,events=load_data(config.data_dir);fit=Snapshot(items,events,config.candidate_cutoff,config.event_weights)
    snap=Snapshot(items,events,config.validation_start,config.event_weights)
    cf=WeightedBPR(4,7).fit(fit.events,epochs=1);codes=encoders(fit.items)
    _,_,groups,meta=build_queries(snap,events,config.test_start,cf,codes,config,False)
    truths,_=relevance(snap,events,config.test_start,config)
    assert truths and sum(groups)==len(meta)
    for user,g in meta.groupby('user_id'):
        candidates,_,_=retrieve(snap,cf,user,config)
        assert set(g.item_id)<=set(candidates)
