from rectools.models.sasrec import SASRecModel
from rectools.models.bert4rec import BERT4RecModel
from .rectools_model import RectoolsRecommender
from pytorch_lightning import Trainer
import typing as tp

from rectools.models.base import ModelConfig

# SASREC_DEFAULT_PARAMS = {
#     "session_max_len": 50,
#     "n_heads": 2,
#     "n_factors": 64,
#     "n_blocks": 2,
#     "lr": 0.001,
#     "loss": "softmax",
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

class RectoolsSASRec(RectoolsRecommender):
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


class RectoolsBERT4Rec(RectoolsRecommender):
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
