"""No external dataset, weights or personally identifying information."""
import csv
import math
import random
from datetime import datetime,timedelta,timezone
from pathlib import Path
from .core import save_json

CATEGORIES=['electronics','home','outdoors','fashion','books']
def generate(output,users=60,items=100,events=4000,seed=42,popularity_bias=1.0):
    if users<10 or items<20 or events<100: raise ValueError('Use >=10 users, >=20 items, >=100 events')
    if popularity_bias<0: raise ValueError('popularity_bias must be nonnegative')
    root=Path(output)
    if (root/'interactions.csv').exists(): raise ValueError('Dataset exists; choose another output')
    root.mkdir(parents=True,exist_ok=True);rng=random.Random(seed);start=datetime(2026,1,1,tzinfo=timezone.utc)
    user_rows=[];item_rows=[]
    for i in range(users):
        cold=i>=int(users*.85);born=start+timedelta(days=35 if cold else 0)
        user_rows.append({'user_id':f'user-{i:04d}','created_at':born.isoformat(),'preferred_category':rng.choice(CATEGORIES),
            'price_sensitivity':round(rng.uniform(.3,1.5),3),'activity_level':round(rng.uniform(.3,2),3),
            'region':rng.choice(['US','EU','APAC'])})
    intrinsic=[]
    for i in range(items):
        born=start+timedelta(days=32 if i>=int(items*.9) else -rng.randrange(1,90))
        item_rows.append({'item_id':f'item-{i:04d}','category':CATEGORIES[i%len(CATEGORIES)],'brand':f'brand-{i%8}',
            'price':round(rng.uniform(10,180),2),'created_at':born.isoformat(),'available':True})
        intrinsic.append((1/(i+1)**.55)**popularity_bias)
    rows=[];previous={}
    for n in range(events):
        timestamp=start+timedelta(seconds=int((n+.5)/events*50*86400))
        eligible_users=[u for u in user_rows if datetime.fromisoformat(u['created_at'])<=timestamp]
        user=rng.choices(eligible_users,weights=[u['activity_level'] for u in eligible_users])[0]
        eligible=[i for i,x in enumerate(item_rows) if datetime.fromisoformat(x['created_at'])<=timestamp]
        weights=[]
        for i in eligible:
            item=item_rows[i];affinity=3.5 if item['category']==user['preferred_category'] else .7
            price=math.exp(-item['price']/180*user['price_sensitivity'])
            age=(timestamp-datetime.fromisoformat(item['created_at'])).total_seconds()/86400
            freshness=1+.7*math.exp(-age/12)
            repeat=1.8 if previous.get(user['user_id'])==item['item_id'] else 1.
            weights.append(intrinsic[i]*affinity*price*freshness*repeat+.025)
        i=rng.choices(eligible,weights=weights)[0];item=item_rows[i];previous[user['user_id']]=item['item_id']
        affinity=item['category']==user['preferred_category'];chance=rng.random()
        event='purchase' if chance<(.1 if affinity else .035) else 'add_to_cart' if chance<.2 else 'click' if chance<.52 else 'view'
        rows.append({'interaction_id':f'event-{n:07d}','user_id':user['user_id'],'item_id':item['item_id'],
                     'timestamp':timestamp.isoformat(),'event_type':event})
    for name,records in [('users',user_rows),('items',item_rows),('interactions',rows)]:
        with (root/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    save_json(root/'dataset.json',{'version':'synthetic-ecommerce-v1','seed':seed,'users':users,'items':items,'events':events,
                                 'popularity_bias':popularity_bias,'days':50,'license':'MIT generated fixtures',
                                 'warning':'Latent preferences are generator-only; ranking derives affinity from past events.'})
    return {'data_dir':str(root),'users':users,'items':items,'events':events}
