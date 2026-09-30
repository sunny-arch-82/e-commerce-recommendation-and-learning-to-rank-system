import math
import pytest
from rankforge.metrics import ranking_metrics

def test_perfect_ranking():
    result=ranking_metrics(['a','b','c'],{'a':3,'b':2},2)
    assert result['recall']==result['precision']==result['map']==result['ndcg']==result['mrr']==1

def test_misses_and_denominators():
    result=ranking_metrics(['x','a'],{'a':2,'b':2},3)
    assert result['precision']==pytest.approx(1/3)
    assert result['recall']==.5 and result['map']==.25 and result['mrr']==.5
    assert result['ndcg']==pytest.approx((3/math.log2(3))/(3+3/math.log2(3)))
    assert ranking_metrics([],{'a':2},10)['recall']==0
    assert ranking_metrics(['a'],{'a':1},1) is None
    with pytest.raises(ValueError):ranking_metrics(['a','a'],{'a':2},2)
