from aprec.recommenders.rectools.transformers import RectoolsSASRec, RectoolsBERT4Rec
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.bert4recrepro.b4vae_bert4rec import B4rVaeBert4Rec
import numpy as np


DATASET = "BERT4rec.ml-1m"

USERS_FRACTIONS = [1]
FILTER_SEEN = True
RANDOM_STATE = 32

EPOCHS = 1

def sasrec_rt():
    return RectoolsSASRec(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS)

def bert4rec_rt():
    return RectoolsBERT4Rec(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS)

def b4rvae_bert4rec(epochs=EPOCHS):
    return B4rVaeBert4Rec(epochs=epochs)

RECOMMENDERS = {
    # "sasrec_rt": sasrec_rt,
    # "bert4rec_rt": bert4rec_rt,
    "b4vae_bert4rec": b4rvae_bert4rec,
}

MAX_TEST_USERS=6040

METRICS = [NDCG(10), MRR()]

RECOMMENDATIONS_LIMIT = 100
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

