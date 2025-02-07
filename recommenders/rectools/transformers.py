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
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pathlib import Path
import torch
from rectools.dataset.identifiers import IdMap
from rectools.models.base import ModelConfig
from pytorch_lightning import LightningModule
from rectools.dataset import IdMap
from rectools.models.nn.item_net import IdEmbeddingsItemNet
from rectools.dataset.dataset import DatasetSchema

# These are also RecTools default params, but let's fix them explicitly
SASREC_DEFAULT_PARAMS = {
    "session_max_len": 100,
    "n_heads": 4,
    "n_factors": 256,
    "n_blocks": 2,
    "lr": 0.001,
    "loss": "softmax",
    "recommend_devices": [1],
    "recommend_batch_size": 128,
}

# These are also RecTools default params, but let's fix them explicitly
BERT4REC_DEFAULT_PARAMS = {
    "session_max_len": 100,
    "n_heads": 4,
    "n_factors": 256,
    "n_blocks": 2,
    "lr": 0.001,
    "loss": "softmax",
    "mask_prob": 0.15,
    "recommend_devices": [1],
    "recommend_batch_size": 128,
}

PATIENCE = 50

def get_trainer(epochs, callbacks, min_epochs: tp.Optional[int]=None):
    if min_epochs is None:
        min_epochs = epochs
    return Trainer(
        max_epochs=epochs,
        min_epochs=min_epochs,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=CSVLogger("rectools_logs"),
        accelerator="gpu",
        devices=[1],
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


class RecallCallback(Callback):
    name: str = "recall"

    def __init__(self, k: int, verbose: int = 0) -> None:
        self.k = k
        self.name += f"@{k}"
        self.verbose = verbose

        self.batch_recall_per_users: tp.List[torch.Tensor] = []

    def on_validation_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: tp.Dict[str, torch.Tensor], 
        batch: tp.Dict[str, torch.Tensor], 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        logits = outputs["logits"]
        if logits is None:
            logits = pl_module.torch_model.encode_sessions(batch["x"], pl_module.item_embs)[:, -1, :]
        _, sorted_batch_recos = logits.topk(k=self.k)

        batch_recos = sorted_batch_recos
        targets = batch["y"]

        # assume all users have the same amount of TP
        liked = targets.shape[1]
        tp_mask = torch.stack([torch.isin(batch_recos[uid], targets[uid]) for uid in range(batch_recos.shape[0])])
        recall_per_users = tp_mask.sum(dim=1) / liked

        self.batch_recall_per_users.append(recall_per_users)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        recall = float(torch.concat(self.batch_recall_per_users).mean())
        self.log_dict({self.name: recall}, on_step=False, on_epoch=True, prog_bar=self.verbose > 0)

        self.batch_recall_per_users.clear()
        

class RectoolsTransformer(RectoolsRecommender):
    def get_item_rankings(self):
        
        result = {}
        items_ranking_requests = {
            request.user_id: request.item_ids for request in self.items_ranking_requests
        }
        users = list(items_ranking_requests.keys())
        processed_dataset = self.model.data_preparator.transform_dataset_u2i(self.dataset, users)
        
        recommend_dataloader = self.model.data_preparator.get_dataloader_recommend(processed_dataset, self.model.recommend_batch_size)
        recommend_device = "cuda:1" if torch.cuda.is_available() else "cpu"
        device = torch.device(recommend_device)
        self.model.torch_model.to(device)
        self.model.torch_model.eval()
        with torch.no_grad():
            item_embs = self.model.torch_model.item_model.get_all_embeddings()
            user_embs = []
            for batch in recommend_dataloader:
                batch_embs = self.model.torch_model.encode_sessions(batch["x"].to(device), item_embs)[:, -1, :]
                user_embs.append(batch_embs)
        user_embs = torch.cat(user_embs)
        
        internal_user_ids = processed_dataset.user_id_map.convert_to_internal(users)
        ui_csr_for_filter = processed_dataset.get_user_item_matrix(include_weights=False)
        
        
        for user_ind, user_id in zip(internal_user_ids, users):
            user_emb = user_embs[user_ind]
            candidates = items_ranking_requests[user_id]
            candidate_indexes, missing = processed_dataset.item_id_map.convert_to_internal(candidates, strict=False, return_missing=True)
            scored_candidates = processed_dataset.item_id_map.convert_to_external(candidate_indexes)
            candidate_embs = item_embs[candidate_indexes]
            scores = candidate_embs @ user_emb
            
            if self.filter_seen:
                mask = (
                    torch.from_numpy(
                        ui_csr_for_filter[user_ind, candidate_indexes].toarray()
                    ).to(scores.device)
                    == 1
                ).squeeze()
                scores = torch.masked_fill(scores, mask, float("-inf"))
            
            scores = scores.detach().cpu().numpy()
            user_result =  [tuple(x) for x in zip(scored_candidates, scores)]
            missing_results = [(id, float("-inf")) for id in missing]
            user_result.extend(missing_results)
            user_result.sort(key=lambda x: -x[1])
            result[user_id] = user_result
            del candidate_embs, scores, mask, user_emb
            torch.cuda.empty_cache()
        return result
    
    def update_weights_from_ckpt(self, ckpt_name_start: str):
        ckpt_dir = Path(self.model.fit_trainer.log_dir) / "checkpoints"
        for pth in ckpt_dir.iterdir():
            if pth.name.startswith(ckpt_name_start):
                ckpt_path = pth
        self.model.lightning_model = self.model.lightning_model.__class__.load_from_checkpoint(
            ckpt_path,
            torch_model=self.model.torch_model,
            data_preparator=self.model.data_preparator,
            model_config = self.model.get_config()
        )


# ### -------------- Recall validated -------------- ### #

class RectoolsBERT4RecRecallValidated(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        def get_trainer_bert():
            last_epoch_ckpt = ModelCheckpoint(filename="bert4rec_last_epoch_{epoch}")
            recall_callback = RecallCallback(10, verbose=1)
            max_recall_ckpt = ModelCheckpoint(
                monitor="recall@10",
                mode="max",
                filename="bert4rec_best_recall_{epoch}-{recall@10:.2f}",
            )
            least_val_loss_ckpt = ModelCheckpoint(
                monitor=SASRecModel.val_loss_name,
                mode="min",
                filename="bert4rec_best_val_loss_{epoch}-{val_loss:.2f}",
            )
            early_stopping_recall = EarlyStopping(
                monitor="recall@10",
                mode="max",
                patience=PATIENCE,
            )
            callbacks = [last_epoch_ckpt, recall_callback, least_val_loss_ckpt, max_recall_ckpt, early_stopping_recall]
            return get_trainer(epochs, callbacks, min_epochs=1)
        
        def get_val_mask_func(interactions: pd.DataFrame):
            return leave_one_out_mask_for_users(interactions, val_users = self.val_users)
        
        self.model = BERT4RecModel(epochs=epochs, verbose=1, deterministic=True, get_trainer_func=get_trainer_bert, get_val_mask_func=get_val_mask_func, **BERT4REC_DEFAULT_PARAMS)
    
    def rebuild_model(self):
        super().rebuild_model()
        self.update_weights_from_ckpt("bert4rec_best_recall")
        
        
class RectoolsSASRecRecallValidated(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        def get_trainer_sasrec():
            last_epoch_ckpt = ModelCheckpoint(filename="sasrec_last_epoch_{epoch}")
            recall_callback = RecallCallback(10, verbose=1)
            max_recall_ckpt = ModelCheckpoint(
                monitor="recall@10",
                mode="max",
                filename="sasrec_best_recall_{epoch}-{recall@10:.2f}",
            )
            least_val_loss_ckpt = ModelCheckpoint(
                monitor=SASRecModel.val_loss_name,
                mode="min",
                filename="sasrec_best_val_loss_{epoch}-{val_loss:.2f}",
            )
            early_stopping_recall = EarlyStopping(
                monitor="recall@10",
                mode="max",
                patience=PATIENCE,
            )
            callbacks = [last_epoch_ckpt, recall_callback, least_val_loss_ckpt, max_recall_ckpt, early_stopping_recall]
            return get_trainer(epochs, callbacks, min_epochs=1)
        
        def get_val_mask_func(interactions: pd.DataFrame):
            return leave_one_out_mask_for_users(interactions, val_users = self.val_users)
        
        self.model = SASRecModel(epochs=epochs, verbose=1, deterministic=True, get_trainer_func=get_trainer_sasrec, get_val_mask_func=get_val_mask_func, **SASREC_DEFAULT_PARAMS)
    
    def rebuild_model(self):
        super().rebuild_model()
        self.update_weights_from_ckpt("sasrec_best_recall")



# ### -------------- Not used in benchmarks -------------- ### #


# For training model without validation fold (not used in benchmarks)
class RectoolsSASRec(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        def get_trainer_sasrec():
            callbacks = ModelCheckpoint(filename="sasrec_last_epoch_{epoch}")
            return get_trainer(epochs, callbacks)
        self.model = SASRecModel(epochs=epochs, verbose=1, deterministic=True, get_trainer_func=get_trainer_sasrec, **SASREC_DEFAULT_PARAMS)
        
# For training model without validation fold (not used in benchmarks)
class RectoolsBERT4Rec(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        def get_trainer_ber4rec():
            callbacks = ModelCheckpoint(filename="bert4rec_last_epoch_{epoch}")
            return get_trainer(epochs, callbacks)
        self.model = BERT4RecModel(epochs=epochs, verbose=1, get_trainer_func=get_trainer_ber4rec, deterministic=True, **BERT4REC_DEFAULT_PARAMS)

# For quick validation from saved checkpoint without re-training the model (not used in benchmarks)
class RectoolsSASRecFromCheckpoint(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        self.model = SASRecModel.load_from_checkpoint(self.ckpt)
        
    def rebuild_model(self):
        interactions = pd.DataFrame(self.interactions)
        interactions["datetime"] = interactions.groupby("user_id").cumcount()
        interactions["weight"] = 1
        self.dataset = Dataset.construct(interactions)

# For quick validation from saved checkpoint without re-training the model (not used in benchmarks)
class RectoolsBERT4RecFromCheckpoint(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        self.model = BERT4RecModel.load_from_checkpoint(self.ckpt)
        
    def rebuild_model(self):
        interactions = pd.DataFrame(self.interactions)
        interactions["datetime"] = interactions.groupby("user_id").cumcount()
        interactions["weight"] = 1
        self.dataset = Dataset.construct(interactions)


# For training model validating on val loss instead of recall (not used in benchmarks)
class RectoolsSASRecValidated(RectoolsTransformer):
    
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        
        def get_trainer_sasrec():
            last_epoch_ckpt = ModelCheckpoint(filename="sasrec_last_epoch_{epoch}")
            least_val_loss_ckpt = ModelCheckpoint(
                monitor=SASRecModel.val_loss_name,
                mode="min",
                filename="sasrec_best_val_loss_{epoch}-{val_loss:.2f}",
            )
            early_stopping_val_loss = EarlyStopping(
                monitor=SASRecModel.val_loss_name,
                mode="min",
                patience=PATIENCE,
            )
            callbacks = [last_epoch_ckpt, least_val_loss_ckpt, early_stopping_val_loss]
            return get_trainer(epochs, callbacks, min_epochs=1)
        
        def get_val_mask_func(interactions: pd.DataFrame):
            return leave_one_out_mask_for_users(interactions, val_users = self.val_users)
        
        self.model = SASRecModel(epochs=epochs, verbose=1, deterministic=True, get_trainer_func=get_trainer_sasrec, get_val_mask_func=get_val_mask_func, **SASREC_DEFAULT_PARAMS)
    
    def rebuild_model(self):
        super().rebuild_model()
        self.update_weights_from_ckpt("sasrec_best_val_loss")

# For training model validating on val loss instead of recall (not used in benchmarks)
class RectoolsBERT4RecValidated(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        def get_trainer_bert():
            last_epoch_ckpt = ModelCheckpoint(filename="bert4rec_last_epoch_{epoch}")
            least_val_loss_ckpt = ModelCheckpoint(
                monitor=SASRecModel.val_loss_name,
                mode="min",
                filename="bert4rec_best_val_loss_{epoch}-{val_loss:.2f}",
            )
            early_stopping_val_loss = EarlyStopping(
                monitor=SASRecModel.val_loss_name,
                mode="min",
                patience=PATIENCE,
            )
            callbacks = [last_epoch_ckpt, least_val_loss_ckpt, early_stopping_val_loss]
            return get_trainer(epochs, callbacks, min_epochs=1)
        
        def get_val_mask_func(interactions: pd.DataFrame):
            return leave_one_out_mask_for_users(interactions, val_users = self.val_users)
        
        self.model = BERT4RecModel(epochs=epochs, verbose=1, deterministic=True, get_trainer_func=get_trainer_bert, get_val_mask_func=get_val_mask_func, **BERT4REC_DEFAULT_PARAMS)
    
    def rebuild_model(self):
        super().rebuild_model()
        self.update_weights_from_ckpt("bert4rec_best_val_loss")
