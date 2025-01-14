from rectools.models.sasrec import SASRecModel
from .rectools_model import RectoolsRecommender

class RectoolsSASRec(RectoolsRecommender):
    def _init_model(self):
        self.model = SASRecModel(epochs=1, verbose=1, deterministic=True)
