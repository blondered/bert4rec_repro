from aprec.recommenders.rectools.transformers import RectoolsSASRec, RectoolsBERT4Rec
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.bert4recrepro.b4vae_bert4rec import B4rVaeBert4Rec
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
import numpy as np

from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
# from aprec.losses.bce import BCELoss
# from aprec.losses.mean_ypred_ploss import MeanPredLoss
# from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
# from aprec.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec
# from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
# from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
# from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
# from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
# from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
# from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
# from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
# from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
# from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
# from aprec.recommenders.bert4recrepro.recbole_bert4rec import RecboleBERT4RecRecommender
from aprec.recommenders.bert4recrepro.b4vae_bert4rec import B4rVaeBert4Rec
from aprec.recommenders.lightfm import LightFMRecommender
# from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
# from aprec.recommenders.bert4recrepro.recbole_bert4rec import RecboleBERT4RecRecommender


DATASET = "BERT4rec.ml-1m"

USERS_FRACTIONS = [1]
FILTER_SEEN = True  # TODO: not applied
RANDOM_STATE = 32

EPOCHS = 10

def sasrec_rt():
    return RectoolsSASRec(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS)

def bert4rec_rt():
    return RectoolsBERT4Rec(filter_seen=FILTER_SEEN, random_state=RANDOM_STATE, epochs=EPOCHS)

def b4rvae_bert4rec(epochs=EPOCHS):
    return FilterSeenRecommender(B4rVaeBert4Rec(epochs=epochs))

# requires tensorflow (NOT CHECKED)
# def dnn(model_arch, loss, sequence_splitter, 
#                 val_sequence_splitter=SequenceContinuation, 
#                  target_builder=FullMatrixTargetsBuilder,
#                 optimizer=Adam(),
#                 training_time_limit=3600, metric=KerasNDCG(40), 
#                 max_epochs=10000
#                 ):
#     return DNNSequentialRecommender(train_epochs=max_epochs, loss=loss,
#                                                           model_arch=model_arch,
#                                                           optimizer=optimizer,
#                                                           early_stop_epochs=100,
#                                                           batch_size=128,
#                                                           training_time_limit=training_time_limit,
#                                                           sequence_splitter=sequence_splitter, 
#                                                           targets_builder=target_builder, 
#                                                           val_sequence_splitter = val_sequence_splitter,
#                                                           metric=metric,
#                                                           debug=False
#                                                           )

# vanilla_sasrec  = lambda: dnn(
#             SASRec(max_history_len=HISTORY_LEN, 
#                             dropout_rate=0.2,
#                             num_heads=1,
#                             num_blocks=2,
#                             vanilla=True, 
#                             embedding_size=50,
#                     ),
#             BCELoss(),
#             ShiftedSequenceSplitter,
#             optimizer=Adam(beta_2=0.98),
#             target_builder=lambda: NegativePerPositiveTargetBuilder(HISTORY_LEN), 
#             metric=BCELoss(),
#             )

def recbole_bert4rec(epochs=EPOCHS):
    return RecboleBERT4RecRecommender(epochs=epochs)

RECOMMENDERS = {
    "sasrec_rt": sasrec_rt,
    # "bert4rec_rt": bert4rec_rt,
    # "b4vae_bert4rec": b4rvae_bert4rec,
    # "vanilla_sasrec": vanilla_sasrec,  # tensorflow required, as well as for original bert4rec
    # "recbole_bert4rec": recbole_bert4rec, # recbole neded, not tested
    # "our_bert4rec": # tensorflow required
}

MAX_TEST_USERS=6040

METRICS = [NDCG(10), MRR()]

RECOMMENDATIONS_LIMIT = 100
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)