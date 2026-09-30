"""Weighted BPR: observed positives vs sampled unobserved items, not explicit dislikes."""
import numpy as np
from scipy.sparse import csr_matrix

class WeightedBPR:
    def __init__(self,factors=16,seed=42):
        self.factors=factors;self.seed=seed;self.user_ids=[];self.item_ids=[];self.user_index={};self.item_index={}
        self.user_factors=np.empty((0,factors));self.item_factors=np.empty((0,factors));self.history={};self.loss=[]
    def matrix(self,events):
        self.user_ids=sorted(events.user_id.unique());self.item_ids=sorted(events.item_id.unique())
        self.user_index={u:i for i,u in enumerate(self.user_ids)};self.item_index={x:i for i,x in enumerate(self.item_ids)}
        return csr_matrix((events.weight.to_numpy(),(events.user_id.map(self.user_index),events.item_id.map(self.item_index))),
                          shape=(len(self.user_ids),len(self.item_ids)))
    def fit(self,events,epochs=5,learning_rate=.03,regularization=.01):
        matrix=self.matrix(events)
        if matrix.shape[0]==0 or matrix.shape[1]<2: raise ValueError('BPR needs users and at least two observed items')
        rng=np.random.default_rng(self.seed)
        self.user_factors=rng.normal(0,.1,(matrix.shape[0],self.factors));self.item_factors=rng.normal(0,.1,(matrix.shape[1],self.factors))
        users=[];choices={};self.history={}
        for u in range(matrix.shape[0]):
            row=matrix.getrow(u);seen=set(row.indices.tolist());self.history[self.user_ids[u]]=sorted(seen)
            if 0<len(seen)<matrix.shape[1]:
                users.append(u);choices[u]=(row.indices,row.data/row.data.sum(),np.array(sorted(set(range(matrix.shape[1]))-seen)))
        if not users: raise ValueError('Every user observed all items; no unobserved pairs for BPR')
        for _ in range(epochs):
            loss=0.;steps=max(matrix.nnz,1)
            for _ in range(steps):
                u=int(rng.choice(users));indices,weights,unobserved=choices[u]
                i=int(rng.choice(indices,p=weights));j=int(rng.choice(unobserved))
                user=self.user_factors[u].copy();pos=self.item_factors[i].copy();neg=self.item_factors[j].copy()
                margin=float(user@(pos-neg));gradient=1/(1+np.exp(np.clip(margin,-35,35)))
                self.user_factors[u]+=learning_rate*(gradient*(pos-neg)-regularization*user)
                self.item_factors[i]+=learning_rate*(gradient*user-regularization*pos)
                self.item_factors[j]+=learning_rate*(-gradient*user-regularization*neg)
                loss+=float(np.logaddexp(0,-margin))
            self.loss.append(loss/steps)
        return self
    def scores(self,user,items):
        if user not in self.user_index: return np.zeros(len(items))
        u=self.user_factors[self.user_index[user]]
        return np.array([float(u@self.item_factors[self.item_index[i]]) if i in self.item_index else 0. for i in items])
    def save(self,path):
        np.savez_compressed(path,user_factors=self.user_factors,item_factors=self.item_factors,
                            user_ids=np.array(self.user_ids,dtype=str),item_ids=np.array(self.item_ids,dtype=str))
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            obj=cls(data['item_factors'].shape[1]);obj.user_ids=data['user_ids'].tolist();obj.item_ids=data['item_ids'].tolist()
            obj.user_factors=data['user_factors'];obj.item_factors=data['item_factors']
        obj.user_index={u:i for i,u in enumerate(obj.user_ids)};obj.item_index={i:n for n,i in enumerate(obj.item_ids)}
        return obj
