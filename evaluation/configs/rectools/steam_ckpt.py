from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.rectools.transformers import RectoolsSASRecValidated, RectoolsBERT4RecValidated
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.recall import Recall
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.recommenders.rectools.transformers import RectoolsSASRecFromCheckpoint, RectoolsBERT4RecFromCheckpoint

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

sasrec_ckpt="rectools_logs/lightning_logs/version_0/checkpoints/sasrec_best_recall_epoch=4-recall@10=0.10.ckpt"
bert4rec_ckpt="rectools_logs/lightning_logs/version_1/checkpoints/bert4rec_best_recall_epoch=14-recall@10=0.10.ckpt"

def sasrec_rt_ckpt():
    return RectoolsSASRecFromCheckpoint(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS, ckpt=sasrec_ckpt)

def bert4rec_rt_ckpt():
    return RectoolsBERT4RecFromCheckpoint(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS, ckpt=bert4rec_ckpt)

RECOMMENDERS = {
    # "sasrec_rt_ckpt": sasrec_rt_ckpt,
    "bert4rec_rt": bert4rec_rt_ckpt,
}
