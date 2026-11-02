"""Self-contained saved model for one case (Deliverable 5) and the API Part B/C use.

    from leakage_ml.predictor import LeakagePredictor
    nmos_off = LeakagePredictor.load("saved_models/NMOS_OFF.joblib")
    currents = nmos_off.predict(df)   # df columns: vdd,temp,w,vd,vs,vg,vb,vth0,toxe,ndep,u0,xj
"""
import joblib
import pandas as pd

from . import config as C
from .data import build_features


class LeakagePredictor:
    """Bundles everything needed to go from raw SPICE-style inputs to amperes:
    feature engineering, dropped columns, the fitted model and the target transform."""

    def __init__(self, case, family, model, target_tf, feature_cols, feature_set, meta=None):
        self.case, self.family, self.model = case, family, model
        self.target_tf, self.feature_cols, self.feature_set = target_tf, feature_cols, feature_set
        self.mos = C.CASES[case]["mos"]
        self.meta = meta or {}

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={"vtho": "vth0"})
        return build_features(df, self.mos, self.feature_set)[self.feature_cols]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Terminal currents in amperes, columns id, ig, is, ib (SPICE sign convention)."""
        y = self.model.predict(self.features(df))
        return pd.DataFrame(self.target_tf.inverse_transform(y), columns=C.TARGETS, index=df.index)

    def save(self, path):
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "LeakagePredictor":
        return joblib.load(path)
