from rectools.models.sasrec import SASRecModel
from .rectools_model import RectoolsRecommender
from pytorch_lightning import Trainer

SASREC_PARAMS = {
    session_max_len: 50,
    n_heads: 2,
    n_factors: 64,
    n_blocks: 2,
    lr: 0.001,
    loss: "softmax"
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
        self.model = SASRecModel(epochs=epochs, verbose=1, deterministic=True, trainer=trainer, **SASREC_PARAMS)
