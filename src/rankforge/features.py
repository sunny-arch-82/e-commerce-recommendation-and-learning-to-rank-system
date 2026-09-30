import numpy as np
import pandas as pd

BASIC=['log_item_popularity','price','item_age_days','category_code','brand_code','user_category_affinity']
RICH=['log_user_events','log_user_purchases','user_recency_days','user_mean_purchase_price','item_purchase_rate',
      'log_pair_events','pair_recency_days','price_relative_to_user','category_popularity','cold_item']

def names(config): return BASIC+(RICH if config.feature_set=='rich' else [])+(['cf_score'] if config.include_cf_score else [])

def encoders(items):
    return {column:{value:i for i,value in enumerate(sorted(items[column].unique()))} for column in ['category','brand']}

def features(snapshot,cf,user,item_ids,codes,config):
    history=snapshot.history(user);affinity=snapshot.affinity(user)
    purchases=history[history.event_type=='purchase'];prices=[snapshot.item_map[i]['price'] for i in purchases.item_id if i in snapshot.item_map]
    mean_price=float(np.mean(prices)) if prices else float(snapshot.items.price.mean()) if len(snapshot.items) else 0.
    cf_scores=cf.scores(user,item_ids);rows=[]
    for item,cf_score in zip(item_ids,cf_scores):
        metadata=snapshot.item_map[item];pair=history[history.item_id==item]
        user_recency=(snapshot.as_of-history.timestamp.max()).total_seconds()/86400 if len(history) else 365.
        pair_recency=(snapshot.as_of-pair.timestamp.max()).total_seconds()/86400 if len(pair) else 365.
        age=(snapshot.as_of-metadata['created_at']).total_seconds()/86400
        category=metadata['category'];item_count=snapshot.item_counts.get(item,0)
        row={'log_item_popularity':np.log1p(snapshot.popularity.get(item,0)),'price':float(metadata['price']),
            'item_age_days':float(age),'category_code':codes['category'].get(category,-1),'brand_code':codes['brand'].get(metadata['brand'],-1),
            'user_category_affinity':affinity.get(category,0.),'log_user_events':np.log1p(len(history)),
            'log_user_purchases':np.log1p(len(purchases)),'user_recency_days':float(user_recency),
            'user_mean_purchase_price':mean_price,'item_purchase_rate':snapshot.item_purchases.get(item,0)/(item_count+5),
            'log_pair_events':np.log1p(len(pair)),'pair_recency_days':float(pair_recency),
            'price_relative_to_user':float(abs(metadata['price']-mean_price)/(mean_price+1)),
            'category_popularity':float(snapshot.popularity.get(item,0))*affinity.get(category,0.),
            'cold_item':int(item not in cf.item_index),'cf_score':float(cf_score)}
        rows.append(row)
    return pd.DataFrame(rows,columns=names(config)).astype(float)
