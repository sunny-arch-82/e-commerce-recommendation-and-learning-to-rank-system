import importlib.metadata
from pathlib import Path
import pandas as pd
from .core import utc,save_json
from .data import load_data,audit,Snapshot
from .collaborative import WeightedBPR
from .features import encoders,names
from .ranking import build_queries,fit_ranker
from .artifacts import write_artifacts

def train(config):
    root=Path(config.output_dir)
    if (root/'checksums.json').exists(): raise ValueError('Run already exists; choose a new output_dir')
    root.mkdir(parents=True,exist_ok=True);quality=audit(config.data_dir,config);save_json(root/'data_quality.json',quality)
    _,items,events=load_data(config.data_dir)
    train_snapshot=Snapshot(items,events,config.candidate_cutoff,config.event_weights)
    validation_snapshot=Snapshot(items,events,config.validation_start,config.event_weights)
    cf=WeightedBPR(config.factors,config.seed).fit(train_snapshot.events,config.bpr_epochs,
        config.bpr_learning_rate,config.bpr_regularization)
    cf.save(root/'candidate.npz');save_json(root/'candidate_training.json',{'pairwise_loss':cf.loss,
        'users':len(cf.user_ids),'items':len(cf.item_ids),'fit_before':train_snapshot.as_of.isoformat()})
    # Encoding vocabularies are frozen at candidate cutoff; future categories map to -1.
    codes=encoders(train_snapshot.items)
    X,y,groups,meta=build_queries(train_snapshot,events,config.validation_start,cf,codes,config,training=True)
    V,vy,vg,_=build_queries(validation_snapshot,events,config.test_start,cf,codes,config,training=False)
    rank_rows=pd.concat([meta,X],axis=1);rank_rows['relevance']=y;rank_rows.to_csv(root/'ranking_train.csv',index=False)
    save_json(root/'ranking_groups.json',{'training':groups,'validation':vg,'training_rows':len(X),'validation_rows':len(V)})
    booster=fit_ranker(X,y,groups,V,vy,vg,config)
    feature_importance={name:float(gain) for name,gain in zip(names(config),booster.feature_importance(importance_type='gain'))}
    save_json(root/'feature_importance.json',{'gain':feature_importance,'note':'Native split gain is not causal evidence'})
    serving_as_of=utc(config.test_start)
    # Fixed test-origin snapshot: only events already observed before test start are persisted.
    history=events[events.timestamp<serving_as_of].copy()
    serving_items=items[items.created_at<serving_as_of].copy()
    metadata={'format_version':1,'config':config.model_dump(),'dataset_hash':quality['dataset_hash'],
        'encoders':codes,'features':names(config),'serving_as_of':serving_as_of.isoformat(),
        'candidate_fit_before':train_snapshot.as_of.isoformat(),'ranker_label_window':[config.candidate_cutoff,config.validation_start],
        'early_stopping_window':[config.validation_start,config.test_start],
        'best_iteration':booster.current_iteration(),
        'environment':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','lightgbm','rankforge']}}
    version=write_artifacts(root,cf,booster,serving_items,history,metadata)
    save_json(root/'run.json',{'model_version':version,'config':config.model_dump(),'metadata':metadata})
    return {'artifacts':str(root),'model_version':version,'ranking_queries':len(groups),'best_iteration':booster.current_iteration(),
            'test_evaluated':False}
