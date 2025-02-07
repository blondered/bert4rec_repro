from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.rectools.common_benchmark_config import *

DATASET = "ml-20m"
MAX_TEST_USERS=138493
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

N_VAL_USERS=1024

USERS_FRACTIONS = [1]
FILTER_SEEN = True
RANDOM_STATE = 32

METRICS = [NDCG(10), Recall(10), MRR()]
RECOMMENDATIONS_LIMIT = 100
# TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)

EPOCHS = 200

def sasrec_val_rt():
    return RectoolsSASRecValidated(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS)

def bert4rec_val_rt():
    return RectoolsBERT4RecValidated(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS)

RECOMMENDERS = {
    "sasrec_val_rt": sasrec_val_rt,
    "bert4rec_val_rt": bert4rec_val_rt,
}

