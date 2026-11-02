"""Target transform for currents that span ~12 decades and can change sign."""
import numpy as np

from . import config as C


class SignedLogTransform:
    """y = sign(I) * log10(1 + |I| / I0), with one I0 per target fitted on TRAIN.

    For |I| >> I0 this is sign(I) * log10(|I| / I0), so an absolute error dy in
    y is a relative current error of about ln(10) * dy: minimising MSE in y
    is (approximately) minimising relative error, which is what the spec scores.
    """

    def __init__(self, frac: float = C.I0_FRAC):
        self.frac = frac
        self.i0_ = None

    def fit(self, I: np.ndarray):
        I = np.atleast_2d(np.asarray(I, dtype=float).T).T
        self.i0_ = self.frac * np.percentile(np.abs(I), 1, axis=0)
        self.i0_ = np.maximum(self.i0_, 1e-30)
        return self

    def transform(self, I: np.ndarray) -> np.ndarray:
        I = np.asarray(I, dtype=float)
        return np.sign(I) * np.log10(1.0 + np.abs(I) / self.i0_)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.sign(y) * (np.power(10.0, np.abs(y)) - 1.0) * self.i0_
