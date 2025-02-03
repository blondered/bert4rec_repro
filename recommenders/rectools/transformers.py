from rectools.models import SASRecModel
from rectools.models import BERT4RecModel
from rectools.dataset import Dataset
from .rectools_model import RectoolsRecommender
from rectools import Columns
from pytorch_lightning import Trainer
import typing as tp
import numpy as np
import pandas as pd
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from rectools.models.base import ModelConfig

SASREC_DEFAULT_PARAMS = {
    "session_max_len": 100,
    "n_heads": 4,
    "n_factors": 256,
    "n_blocks": 2,
    "lr": 0.001,
    "loss": "softmax",
}

BERT4REC_DEFAULT_PARAMS = {
    "session_max_len": 100,
    "n_heads": 4,
    "n_factors": 256,
    "n_blocks": 2,
    "lr": 0.001,
    "loss": "softmax",
    "mask_prob": 0.15   
}

RectoolsTransformer = RectoolsRecommender

# NOT WORKING
# class RectoolsTransformer(RectoolsRecommender):
#     def get_item_rankings(self):
#         result = {}
#         item_embs = self.model.lightning_model.item_embs.detach().cpu().numpy()
#         users = [request.user_id for request in self.items_ranking_requests]
#         processed_dataset = self.model.data_preparator.transform_dataset_u2i(self.dataset, users)
#         recommend_trainer = Trainer(devices=1, accelerator=self.model.recommend_device)
#         recommend_dataloader = self.model.data_preparator.get_dataloader_recommend(processed_dataset)
#         session_embs = recommend_trainer.predict(model=self.model.lightning_model, dataloaders=recommend_dataloader)
#         user_embs = np.concatenate(session_embs, axis=0)

#         user_inds = processed_dataset.user_id_map.convert_to_internal(users)

#         for user_ind, user_id in zip(user_inds, users): 
#             user_emb = user_embs[user_ind]
#             candidates = self.items_ranking_requests[user_ind].item_ids
#             candidate_indexes, missing = self.model.data_preparator.item_id_map.convert_to_internal(candidates, strict=False, return_missing=True)
#             scored_candidates = [x for x in candidates if x not in missing]
#             candidate_embs = item_embs[candidate_indexes]
#             scores = candidate_embs @ user_emb
#             user_result =  [tuple(x) for x in zip(scored_candidates, scores)]
#             missing_results = [(id, float("-inf")) for id in missing]
#             user_result.extend(missing_results)
#             user_result.sort(key=lambda x: -x[1])
#             result[user_id] = user_result
#         return result


def get_trainer(epochs, callbacks):
    return Trainer(
        max_epochs=epochs,
        min_epochs=epochs,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=CSVLogger("rectools_logs"),
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
    )


def leave_one_out_mask_for_users(interactions: pd.DataFrame, val_users) -> np.ndarray:
    rank = (
        interactions
        .sort_values(Columns.Datetime, ascending=False, kind="stable")
        .groupby(Columns.User, sort=False)
        .cumcount()
    )
    val_mask = (
        (interactions[Columns.User].isin(val_users))
        & (rank == 0)
    )
    return val_mask.values
    
    
    
class RectoolsSASRecValidated(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        
        def get_trainer_sasrec():
            last_epoch_ckpt = ModelCheckpoint(filename="sasrec_last_epoch_{epoch}")
            least_val_loss_ckpt = ModelCheckpoint(
                monitor=SASRecModel.val_loss_name,   # or just pass "val_loss" here,
                mode="min",
                filename="sasrec_best_val_loss_{epoch}-{val_loss:.2f}",
            )
            early_stopping_val_loss = EarlyStopping(
                monitor=SASRecModel.val_loss_name,   # or just pass "val_loss" here
                mode="min",
                patience=20,
            )
            callbacks = [last_epoch_ckpt, least_val_loss_ckpt, early_stopping_val_loss]
            return get_trainer(epochs, callbacks)
        
        def get_val_mask_func(interactions: pd.DataFrame):
            return leave_one_out_mask_for_users(interactions, val_users = self.val_users)
        
        self.model = SASRecModel(epochs=epochs, verbose=1, deterministic=True, get_trainer_func=get_trainer_sasrec, get_val_mask_func=get_val_mask_func, **SASREC_DEFAULT_PARAMS)



class RectoolsSASRec(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        
        def get_trainer_sasrec():
            callbacks = ModelCheckpoint(filename="sasrec_last_epoch_{epoch}")
            return get_trainer(epochs, callbacks)
    
        
        self.model = SASRecModel(epochs=epochs, verbose=1, deterministic=True, get_trainer_func=get_trainer_sasrec, **SASREC_DEFAULT_PARAMS)

# For quick validation from saved checkpoint without re-training the model
class RectoolsSASRecFromCheckpoint(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        self.model = SASRecModel.load_from_checkpoint(self.ckpt)
    def rebuild_model(self):
        interactions = pd.DataFrame(self.interactions)
        interactions["datetime"] = interactions.groupby("user_id").cumcount()
        interactions["weight"] = 1
        self.dataset = Dataset.construct(interactions)

class RectoolsBERT4RecFromCheckpoint(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        self.model = BERT4RecModel.load_from_checkpoint(self.ckpt)
    def rebuild_model(self):
        interactions = pd.DataFrame(self.interactions)
        interactions["datetime"] = interactions.groupby("user_id").cumcount()
        interactions["weight"] = 1
        self.dataset = Dataset.construct(interactions)
        

class RectoolsBERT4Rec(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        def get_trainer_ber4rec():
            callbacks = ModelCheckpoint(filename="bert4rec_last_epoch_{epoch}")
            return get_trainer(epochs, callbacks)
        self.model = BERT4RecModel(epochs=epochs, verbose=1, get_trainer_func=get_trainer_ber4rec, deterministic=True, **BERT4REC_DEFAULT_PARAMS)
