from rectools.models.sasrec import SASRecModel
from .rectools_model import RectoolsRecommender
from pytorch_lightning import Trainer

class RectoolsSASRec(RectoolsRecommender):
    def _init_model(self):
        trainer = Trainer(
            max_epochs=1,
            min_epochs=1,
            deterministic=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            logger=True,
            accelerator="cpu"  # TODO: change
        )
        self.model = SASRecModel(epochs=1, verbose=1, deterministic=True, trainer=trainer)
