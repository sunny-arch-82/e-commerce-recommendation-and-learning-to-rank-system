from fastapi.testclient import TestClient
from rankforge.api import app

def test_not_ready(monkeypatch):
    monkeypatch.delenv('RANKFORGE_ARTIFACTS',raising=False)
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        assert client.get('/ready').status_code==503

def test_recommend_contract(artifacts,monkeypatch):
    monkeypatch.setenv('RANKFORGE_ARTIFACTS',str(artifacts))
    with TestClient(app) as client:
        assert client.get('/ready').status_code==200
        result=client.get('/recommend/new-user?k=5')
        assert result.status_code==200
        response=result.json();assert response['user_id']=='new-user' and response['strategy']=='global_popularity'
        assert all(r['rank']>0 and 'candidate_score' in r for r in response['recommendations'])
        assert client.get('/recommend/new-user?k=0').status_code==422
