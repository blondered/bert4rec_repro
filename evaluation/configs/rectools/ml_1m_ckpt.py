from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.rectools.common_benchmark_config import *
from aprec.recommenders.rectools.transformers import RectoolsSASRecFromCheckpoint

DATASET = "BERT4rec.ml-1m"
MAX_TEST_USERS = 6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

# N_VAL_USERS=2048 
# Doesn't affect neither of the models
# B4rVaeBert4Rec always pops one last action for validation
# Rectools does no validation

def sasrec_rt_ckpt():
    return RectoolsSASRecFromCheckpoint(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS, ckpt="SASRecModel_ml1m")


RECOMMENDERS = {
    "sasrec_rt_ckpt": sasrec_rt_ckpt,
    # "bert4rec_rt": bert4rec_rt,
}