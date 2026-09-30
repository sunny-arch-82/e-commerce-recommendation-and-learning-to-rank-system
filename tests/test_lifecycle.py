import json
import pytest
from rankforge.inference import Recommender
from rankforge.evaluation import evaluate
from rankforge.artifacts import load_artifacts

def test_training_serving_evaluation(artifacts,tmp_path):
    model=Recommender(artifacts)
    for user in [model.cf.user_ids[0],'new-user']:
        result=model.recommend(user,5)
        assert len(result.recommendations)<=5 and result.model_version
        assert len({r.item_id for r in result.recommendations})==len(result.recommendations)
    with pytest.raises(ValueError):model.recommend('new-user',model.config.candidate_count+1)
    report=evaluate(artifacts,tmp_path/'eval','test')
    assert report['queries']>0
    data=json.loads((tmp_path/'eval/metrics.json').read_text())
    assert {m['method'] for m in data['metrics']}=={'popularity','category','cf','ranker'}

def test_corruption_rejected(artifacts,tmp_path):
    import shutil
    from pathlib import Path
    copy=tmp_path/'copy';shutil.copytree(artifacts,copy)
    p=Path(copy/'metadata.json');p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='checksum'):load_artifacts(copy)
