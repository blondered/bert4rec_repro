from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.rectools.common_benchmark_config import *

DATASET = "ml-20m"
MAX_TEST_USERS=138493
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

# N_VAL_USERS=1024
# Doesn't affect neither of the models
# B4rVaeBert4Rec always pops one last action for validation
# Rectools does no validation
