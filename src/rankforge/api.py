import os
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Query
from .inference import Recommender,Response

@asynccontextmanager
async def lifespan(app):
    root=os.getenv('RANKFORGE_ARTIFACTS');app.state.recommender=Recommender(root) if root else None
    yield
app=FastAPI(title='RankForge',lifespan=lifespan)
@app.get('/health')
def health():return {'healthy':True}
@app.get('/ready')
def ready():
    if app.state.recommender is None:raise HTTPException(503,'Set RANKFORGE_ARTIFACTS to a trained artifact directory')
    return {'ready':True,'model_version':app.state.recommender.version}
@app.get('/recommend/{user_id}',response_model=Response)
def recommend(user_id:str,k:int=Query(10,ge=1,le=1000),category:str|None=None):
    if app.state.recommender is None:raise HTTPException(503,'Model unavailable')
    try:return app.state.recommender.recommend(user_id,k,category=category)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
