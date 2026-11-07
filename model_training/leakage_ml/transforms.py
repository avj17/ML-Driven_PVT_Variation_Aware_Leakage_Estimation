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
        # Percentile of the non-zero currents: some body currents are exactly 0 (PMOS_OFF Ib, new data).
        nonzero = np.where(I != 0, np.abs(I), np.nan)
        self.i0_ = self.frac * np.nanpercentile(nonzero, 1, axis=0)
        self.i0_ = np.maximum(self.i0_, 1e-30)
        return self

    def transform(self, I: np.ndarray) -> np.ndarray:
        I = np.asarray(I, dtype=float)
        return np.sign(I) * np.log10(1.0 + np.abs(I) / self.i0_)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.sign(y) * (np.power(10.0, np.abs(y)) - 1.0) * self.i0_


class KCLTransform:
    """Predict [A, Ig, Ib] instead of [Id, Ig, Is, Ib], where A = (Id - Is)/2.

    By KCL the symmetric part (Id + Is)/2 equals S = -(Ig + Ib)/2, so
    Id = A + S and Is = -A + S are rebuilt exactly and the four predicted
    currents always satisfy Id + Ig + Is + Ib = 0. In the OFF state Id and Is
    are near-cancellations of A and S (which are each smooth with a fixed sign
    pattern), so learning A, Ig and Ib is easier than learning Id and Is
    directly. Each of the three is stored in signed-log form (SignedLogTransform).
    """

    def fit(self, I: np.ndarray):
        self.sl_ = SignedLogTransform().fit(self._parts(I))
        return self

    @staticmethod
    def _parts(I):
        I = np.asarray(I, dtype=float)
        return np.column_stack([(I[:, 0] - I[:, 2]) / 2, I[:, 1], I[:, 3]])

    def transform(self, I: np.ndarray) -> np.ndarray:
        return self.sl_.transform(self._parts(I))

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        a, ig, ib = self.sl_.inverse_transform(z).T
        s = -(ig + ib) / 2
        return np.column_stack([a + s, ig, -a + s, ib])
