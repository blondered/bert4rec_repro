from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.rectools.rectools_model import RectoolsRecommender
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.split_actions import LeaveOneOut
import numpy as np


DATASET = "BERT4rec.ml-1m"

USERS_FRACTIONS = [1]
FILTER_SEEN = True
RANDOM_STATE = 32

def top_recommender_rt():
    config = {
        "cls": "PopularModel", 
        "popularity": "n_interactions"
    }
    return RectoolsRecommender(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, model_config=config)

def lightfm_recommender_rt(no_components, loss):
    config = {
        "cls": "LightFMWrapperModel",
        "model": {
            "no_components": no_components,
            "random_state": 32,
            "loss": loss
        },
        "epochs": 20,
    }
    return RectoolsRecommender(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, model_config=config)

RECOMMENDERS = {
    "top_recommender_rt": top_recommender_rt,
    "lightfm_recommender_rt": lambda: lightfm_recommender_rt(30, 'bpr'),
}

MAX_TEST_USERS=6040

METRICS = [NDCG(10), MRR()]

RECOMMENDATIONS_LIMIT = 100
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

