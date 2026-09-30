import pytest
from rankforge.core import Config
from rankforge.generator import generate
from rankforge.training import train

@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    root=tmp_path_factory.mktemp('catalog')/'data'
    generate(root,users=20,items=40,events=1000,seed=7)
    return root

@pytest.fixture(scope='module')
def config(dataset,tmp_path_factory):
    return Config(data_dir=str(dataset),output_dir=str(tmp_path_factory.mktemp('artifacts')),
        factors=4,bpr_epochs=1,ranker_trees=8,ranker_min_child_samples=2,candidate_count=25,ks=[5,10],negatives_per_query=15)

@pytest.fixture(scope='module')
def artifacts(config):
    # Tiny CPU lifecycle smoke only. Not a recommendation-quality benchmark.
    train(config);return config.output_dir
