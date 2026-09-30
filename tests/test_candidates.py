import numpy as np
from rankforge.data import load_data,Snapshot
from rankforge.collaborative import WeightedBPR
from rankforge.candidates import retrieve,sample_negatives

def fitted(config):
    _,items,events=load_data(config.data_dir);snapshot=Snapshot(items,events,config.candidate_cutoff,config.event_weights)
    cf=WeightedBPR(4,8).fit(snapshot.events,epochs=1)
    return snapshot,cf

def test_sparse_and_scores(config):
    snapshot,cf=fitted(config)
    matrix=cf.matrix(snapshot.events)
    assert matrix.shape==(len(cf.user_ids),len(cf.item_ids)) and matrix.nnz>0
    assert matrix.sum()==snapshot.events.weight.sum()
    scores=cf.scores(cf.user_ids[0],cf.item_ids)
    assert len(scores)==len(cf.item_ids) and np.isfinite(scores).all()

def test_candidate_exclusions_and_cold(config):
    snapshot,cf=fitted(config);user=cf.user_ids[0]
    items,_,_=retrieve(snapshot,cf,user,config)
    purchased=set(snapshot.history(user).query("event_type=='purchase'").item_id)
    assert len(items)==len(set(items)) and not purchased.intersection(items)
    cold,_,strategy=retrieve(snapshot,cf,'unknown-user',config)
    assert cold and strategy=='global_popularity'

def test_negative_sampling():
    eligible=['a','b','c','d'];positive=['a'];candidates=['a','d','c']
    for strategy in ['random','popularity','hard']:
        result=sample_negatives(eligible,positive,candidates,{'b':10},2,strategy,3)
        assert len(result)==len(set(result))==2 and 'a' not in result
        assert result==sample_negatives(eligible,positive,candidates,{'b':10},2,strategy,3)
    assert sample_negatives(eligible,positive,candidates,{},2,'hard',3)==['d','c']

def test_safe_serialization(config,tmp_path):
    _,cf=fitted(config);path=tmp_path/'cf.npz';cf.save(path);restored=WeightedBPR.load(path)
    assert restored.user_ids==cf.user_ids
    np.testing.assert_allclose(restored.scores(cf.user_ids[0],cf.item_ids),cf.scores(cf.user_ids[0],cf.item_ids))
