from rectools.models.sasrec import SASRecModel
from rectools.models.bert4rec import BERT4RecModel
from .rectools_model import RectoolsRecommender
from pytorch_lightning import Trainer
import typing as tp
import numpy as np

from rectools.models.base import ModelConfig


# SASREC_DEFAULT_PARAMS = {
#     "session_max_len": 50,
#     "n_heads": 2,
#     "n_factors": 64,
#     "n_blocks": 2,
#     "lr": 0.001,
#     "loss": "softmax",
# }

# BERT4REC_DEFAULT_PARAMS = {
#     "session_max_len": 50,
#     "n_heads": 2,
#     "n_factors": 64,
#     "n_blocks": 2,
#     "lr": 0.001,
#     "loss": "softmax",
#     "mask_prob": 0.2   
# }

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

class RectoolsTransformer(RectoolsRecommender):

    # NOT WORKING
    def get_item_rankings(self):
        result = {}
        item_embs = self.model.lightning_model.item_embs.detach().cpu().numpy()
        users = [request.user_id for request in self.items_ranking_requests]
        processed_dataset = self.model.data_preparator.transform_dataset_u2i(self.dataset, users)
        recommend_trainer = Trainer(devices=1, accelerator=self.model.recommend_device)
        recommend_dataloader = self.model.data_preparator.get_dataloader_recommend(processed_dataset)
        session_embs = recommend_trainer.predict(model=self.model.lightning_model, dataloaders=recommend_dataloader)
        user_embs = np.concatenate(session_embs, axis=0)

        user_inds = processed_dataset.user_id_map.convert_to_internal(users)

        for user_ind, user_id in zip(user_inds, users):
            user_emb = user_embs[user_ind]
            candidates = self.items_ranking_requests[user_ind].item_ids
            candidate_indexes, missing = self.model.data_preparator.item_id_map.convert_to_internal(candidates, strict=False, return_missing=True)
            scored_candidates = [x for x in candidates if x not in missing]
            candidate_embs = item_embs[candidate_indexes]
            scores = candidate_embs @ user_emb
            user_result =  [tuple(x) for x in zip(scored_candidates, scores)]
            missing_results = [(id, float("-inf")) for id in missing]
            user_result.extend(missing_results)
            user_result.sort(key=lambda x: -x[1])
            result[user_id] = user_result
        return result

class RectoolsSASRec(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        trainer = Trainer(
            max_epochs=epochs,
            min_epochs=epochs,
            deterministic=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=True,
            accelerator="gpu",
            devices=1
        )
        self.model = SASRecModel(epochs=epochs, verbose=1, deterministic=True, trainer=trainer, **SASREC_DEFAULT_PARAMS)


class RectoolsBERT4Rec(RectoolsTransformer):
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        trainer = Trainer(
            max_epochs=epochs,
            min_epochs=epochs,
            deterministic=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=True,
            accelerator="gpu",
            devices=1
        )
        self.model = BERT4RecModel(epochs=epochs, verbose=1, deterministic=True, trainer=trainer, **BERT4REC_DEFAULT_PARAMS)
