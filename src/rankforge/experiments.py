from pathlib import Path
from .core import Config,save_json,config_from
from .training import train
from .evaluation import evaluate

def tune(config,space,trials,output):
    import optuna
    if trials<1:raise ValueError('trials must be positive')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    def objective(trial):
        updates={name:trial.suggest_categorical(name,values) for name,values in space.items()}
        cfg=Config.model_validate({**config.model_dump(),**updates,'output_dir':str(root/f'trial-{trial.number:03d}')})
        train(cfg);report=evaluate(cfg.output_dir,root/f'validation-{trial.number:03d}',split='validation')
        k=min(cfg.ks,key=lambda k:abs(k-10));score=next(m['ndcg'] for m in report['metrics'] if m['method']=='ranker' and m['k']==k)
        trial.set_user_attr('artifacts',cfg.output_dir);return score
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=config.seed),
        storage=f"sqlite:///{(root/'study.db').resolve()}",study_name='rankforge',load_if_exists=False)
    study.optimize(objective,n_trials=trials)
    result={'best_validation_ndcg':study.best_value,'params':study.best_params,'artifacts':study.best_trial.user_attrs['artifacts'],
            'selection_basis':'Validation NDCG at configured K nearest 10; no held-out test evaluation'}
    save_json(root/'best.json',result);return result

def compare(paths,output):
    rows=[];root=Path(output);root.mkdir(parents=True,exist_ok=True)
    for path in paths:
        config=config_from(path);run=train(config);report=evaluate(config.output_dir,root/Path(config.output_dir).name,'validation')
        rows.append({'config':path,'run':run,'validation':report})
    save_json(root/'comparison.json',{'runs':rows,'test_evaluated':False});return rows
