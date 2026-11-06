"""
Five forecasting pipelines. Every one of them gets the same input: returns
that are pure noise. Nothing in the data can predict the future, so any
honest pipeline should score an out-of-sample R^2 of about zero.

Every pipeline forecasts the same thing: the sum of the next H returns.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

H = 20  # forecast horizon: the next 20 returns


def future_sum(r, shift=H):
    # sum of r[t+shift-H+1], ..., r[t+shift]
    # with shift = H that's exactly the next H returns, which is what I want.
    # with shift < H the window slides back and starts overlapping the past.
    s = pd.Series(r).rolling(H).sum()
    return s.shift(-shift).to_numpy()


def past_sum(r):
    # sum of r[t-H+1], ..., r[t]: only things I'd know at time t
    return pd.Series(r).rolling(H).sum().to_numpy()


def oos_r2(x, y, model=None, train_frac=0.7):
    # train on the first 70%, test on the last 30%, with a gap of H
    # so no training target overlaps a test target
    if model is None:
        model = LinearRegression()
    x = np.asarray(x).reshape(len(y), -1)
    ok = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    cut = int(train_frac * len(y))
    model.fit(x[:cut], y[:cut])
    pred = model.predict(x[cut + H:])
    test = y[cut + H:]
    return 1 - np.sum((test - pred) ** 2) / np.sum((test - test.mean()) ** 2)


# ---------------------------------------------------------------- honest
def honest(r):
    return oos_r2(past_sum(r), future_sum(r))


# ---------------------------------------------------------------- leak 1
def overlapping_target(r, shift=10):
    # I shift the target forward by 10 instead of 20, so half of the 20
    # "future" returns are actually already inside the feature window.
    # One wrong number and the model is graded partly on the past.
    return oos_r2(past_sum(r), future_sum(r, shift=shift))


# ---------------------------------------------------------------- leak 2
def centred_smoothing(r, w=21):
    # A smoothed feature looks harmless, but rolling(center=True) uses
    # 10 returns from the past AND 10 from the future.
    smooth = pd.Series(r).rolling(w, center=True).mean().to_numpy()
    return oos_r2(smooth, future_sum(r))


# ---------------------------------------------------------------- leak 3
def _slow_features(r):
    # three slow-moving, past-only features. Nothing wrong with them by
    # themselves: they're just averages of returns I've already seen.
    # NOTE: because they move so slowly, together they act like a
    # timestamp. Nearby days have nearly the same three numbers.
    s = pd.Series(r)
    return np.column_stack([s.rolling(w).mean() for w in (100, 300, 1000)])


def _cv_r2(r, cv):
    x, y = _slow_features(r), future_sum(r)
    ok = ~np.isnan(x).any(axis=1) & ~np.isnan(y)
    model = KNeighborsRegressor(n_neighbors=20)
    return cross_val_score(model, x[ok], y[ok], cv=cv, scoring="r2").mean()


def shuffled_cv(r):
    # Shuffled k-fold: neighbouring days land in train and test. Their
    # targets share 19 of 20 returns and their features are almost
    # identical, so the model "predicts" a test day by remembering
    # the day next to it.
    return _cv_r2(r, KFold(5, shuffle=True, random_state=0))


def time_ordered_cv(r):
    # The fix for leak 3: same model, same features, but train only on
    # the past and leave a gap of H before each test block.
    return _cv_r2(r, TimeSeriesSplit(5, gap=H))


PIPELINES = {
    "honest\n(past → future)": honest,
    "leak 1:\noverlapping target": overlapping_target,
    "leak 2:\ncentred smoothing": centred_smoothing,
    "leak 3:\nshuffled CV": shuffled_cv,
    "leak 3 fixed:\ntime-ordered CV": time_ordered_cv,
}
