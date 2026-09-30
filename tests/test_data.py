import pandas as pd
import pytest
from rankforge.generator import generate
from rankforge.core import utc,EVENTS,Config
from rankforge.data import load_data,audit,Snapshot,relevance

def test_generation_deterministic(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b'
    generate(a,10,20,100,3);generate(b,10,20,100,3)
    for name in ['users.csv','items.csv','interactions.csv']:
        assert (a/name).read_bytes()==(b/name).read_bytes()

def test_validation(config):
    assert audit(config.data_dir,config)['passed']
    with pytest.raises(ValueError):Config(test_start='2025-01-01T00:00:00Z')
    with pytest.raises(ValueError):utc('2026-01-01')

def test_snapshot_strict_past(config):
    _,items,events=load_data(config.data_dir)
    snap=Snapshot(items,events,config.candidate_cutoff,config.event_weights)
    assert snap.events.timestamp.max()<utc(config.candidate_cutoff)
    assert snap.items.created_at.max()<snap.as_of
    assert set(snap.events.weight)==set(config.event_weights.values())
    modified=events.copy();future=modified.timestamp>=snap.as_of
    modified.loc[future,'event_type']='purchase'
    other=Snapshot(items,modified,config.candidate_cutoff,config.event_weights)
    assert snap.popularity==other.popularity
    pd.testing.assert_frame_equal(snap.events,other.events)

def test_labels_after_cutoff(config):
    _,items,events=load_data(config.data_dir);snap=Snapshot(items,events,config.candidate_cutoff,config.event_weights)
    truth,_=relevance(snap,events,config.validation_start,config)
    eligible_events=events[(events.timestamp>=snap.as_of)&(events.timestamp<utc(config.validation_start))]
    for user,values in truth.items():
        assert set(values)<=set(eligible_events[eligible_events.user_id==user].item_id)
        assert set(values.values())<=set(EVENTS.values())
