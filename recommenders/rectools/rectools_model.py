import os
import torch
from aprec.recommenders.recommender import Recommender
import typing as tp
from aprec.api.action import Action
import pandas as pd
import numpy as np
from lightning_fabric import seed_everything

from rectools.dataset import Dataset
import threadpoolctl

from rectools.models.base import ModelConfig
from rectools.models import model_from_config


class RectoolsRecommender(Recommender):
    def __init__(self, filter_seen: bool, random_state: int=32, model_config: tp.Optional[ModelConfig]=None, epochs: int=1):
        super().__init__()
        self.interactions = []
        self.filter_seen = filter_seen

        # Enable deterministic behaviour with CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Enable correct multithreading for `implicit` ranker
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        threadpoolctl.threadpool_limits(1, "blas")

        # fix randomness
        torch.use_deterministic_algorithms(True)
        seed_everything(random_state, workers=True)
        
        self._init_model(model_config, epochs)
        
    def _init_model(self, model_config: tp.Optional[ModelConfig], epochs:int = 1):
        if model_config is None:
            raise NotImplementedError
        self.model = model_from_config(model_config)
        

    def add_action(self, action: Action):
        self.interactions.append({"user_id": action.user_id, "item_id": action.item_id})

    def rebuild_model(self):
        interactions = pd.DataFrame(self.interactions)
        interactions["datetime"] = interactions.groupby("user_id").cumcount()
        interactions["weight"] = 1
        self.dataset = Dataset.construct(interactions)
        self.model.fit(self.dataset)

    # recommendation_request: tuple(user_id, features)
    def recommend_batch(self, recommendation_requests, limit):
        user_ids = [request[0] for request in recommendation_requests]
        user_ids_unique = np.array(list(set(user_ids)))
        reco = self.model.recommend(
            users=user_ids_unique,
            dataset=self.dataset,
            filter_viewed=self.filter_seen,
            k=limit
        )
        reco["rec_tuple"] = tuple(zip(reco["item_id"], reco["score"]))
        grouped_reco = reco.groupby("user_id")["rec_tuple"].apply(list)
        sorted_reco = pd.DataFrame({"user_id": user_ids})
        sorted_reco = sorted_reco.merge(grouped_reco, how="left", on="user_id")
        res = [el if isinstance(el, list) else [] for el in list(sorted_reco["rec_tuple"].values)]
        return res

    # this works but too slow
    # cold users are not handled (should raise but no tested)
    # def recommend(self, user_id_external, limit, features=None):
    #     user_reco = self.model.recommend(
    #         users=[user_id_external],
    #         dataset=self.dataset,
    #         filter_viewed=self.filter_seen,
    #         k=limit
    #     )
    #     return tuple(zip(user_reco["item_id"].values, user_reco["score"].values))
