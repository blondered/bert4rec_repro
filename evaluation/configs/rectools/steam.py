from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.rectools.transformers import RectoolsBERT4RecRecallValidated, RectoolsSASRecRecallValidated
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.recall import Recall
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler

DATASET = "BERT4rec.steam"
MAX_TEST_USERS=281428
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)


N_VAL_USERS=2048

USERS_FRACTIONS = [1]
FILTER_SEEN = True
RANDOM_STATE = 32

METRICS = [NDCG(10), Recall(10), MRR()]
RECOMMENDATIONS_LIMIT = 100
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)

EPOCHS = 200

def bert4rec_recall_val_rt():
    return RectoolsBERT4RecRecallValidated(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS)

def sasrec_recall_val_rt():
    return RectoolsSASRecRecallValidated(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS)

RECOMMENDERS = {
    "sasrec_recall_val_rt": sasrec_recall_val_rt,
    "bert4rec_recall_val_rt": bert4rec_recall_val_rt,
}
