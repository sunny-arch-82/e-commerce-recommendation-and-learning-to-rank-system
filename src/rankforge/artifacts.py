import json
from pathlib import Path
import pandas as pd
from .core import Config,save_json,sha,utc
from .collaborative import WeightedBPR

FILES=['candidate.npz','ranker.txt','items.csv','history.csv','metadata.json']
def write_artifacts(root,cf,booster,items,history,metadata):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    cf.save(root/'candidate.npz');booster.save_model(str(root/'ranker.txt'))
    items.to_csv(root/'items.csv',index=False);history.to_csv(root/'history.csv',index=False)
    save_json(root/'metadata.json',metadata)
    checks={name:sha(root/name) for name in FILES}
    from hashlib import sha256
    version=sha256(json.dumps(checks,sort_keys=True).encode()).hexdigest()[:16]
    save_json(root/'checksums.json',{'format_version':1,'model_version':version,'files':checks})
    return version

def load_artifacts(root):
    import lightgbm as lgb
    root=Path(root);checks=json.loads((root/'checksums.json').read_text())
    if checks.get('format_version')!=1 or set(checks['files'])!=set(FILES): raise ValueError('Invalid artifact manifest')
    for name,digest in checks['files'].items():
        if sha(root/name)!=digest: raise ValueError(f'Artifact checksum mismatch: {name}')
    meta=json.loads((root/'metadata.json').read_text());config=Config.model_validate(meta['config'])
    from .features import names
    if meta.get('format_version')!=1 or meta.get('features')!=names(config):
        raise ValueError('Unsupported feature schema')
    cf=WeightedBPR.load(root/'candidate.npz');booster=lgb.Booster(model_file=str(root/'ranker.txt'))
    items=pd.read_csv(root/'items.csv',dtype={'item_id':str});history=pd.read_csv(root/'history.csv',dtype={'user_id':str,'item_id':str,'interaction_id':str})
    items['created_at']=items.created_at.map(utc);history['timestamp']=history.timestamp.map(utc)
    items['available']=items.available.astype(str).str.lower().isin(['true','1'])
    return config,cf,booster,items,history,meta,checks['model_version']
