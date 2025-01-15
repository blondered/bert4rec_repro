from aprec.recommenders.rectools.rectools_sasrec import RectoolsSASRec
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.split_actions import LeaveOneOut
import numpy as np


DATASET = "BERT4rec.ml-1m"

USERS_FRACTIONS = [1]
FILTER_SEEN = True
RANDOM_STATE = 32

EPOCHS = 1

def sasrec_rt():
    return RectoolsSASRec(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS)

RECOMMENDERS = {
    "sasrec_rt": sasrec_rt,
}

MAX_TEST_USERS=6040

METRICS = [NDCG(10), MRR()]

RECOMMENDATIONS_LIMIT = 100
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

