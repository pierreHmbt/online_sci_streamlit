import numpy as np
import pandas as pd
from scipy.optimize import fsolve, brentq

def infoACI_precompute_q_optim(
    dt_Xt,
    dt_Yt,
    dt_Ypredt,
    dt_Scorest,
    info_fun,
    Error_func,
    gamma,
    alpha=0.1,
    q0=1,
    reac0=0,
    B=0,
):
    dates = dt_Yt.index

    X = dt_Xt.to_numpy()
    Y = dt_Yt["value"].to_numpy()
    Ypred = dt_Ypredt["value"].to_numpy()
    Scores = dt_Scorest.to_numpy()
    gamma = np.asarray(gamma)

    T = len(Y)

    Qt = np.empty(T)
    errt = np.empty(T)
    infot = np.empty(T)

    Qt[0] = q0
    errt[0] = np.nan
    infot[0] = np.nan

    ll = 0

    for r in range(1, T):
        q_prev = Qt[r - 1]

        err_t = Error_func(Ypred[r], Y[r], X[r], Scores[r], q_prev)
        reac_t = err_t if q_prev >= reac0 else 1.0

        if info_fun(Ypred[r], Y[r], X[r], Scores[r], q_prev):
            ll += 1
            infot[r] = 1.0
            qt_new = q_prev + gamma[ll - 1] * (reac_t - alpha)
        else:
            infot[r] = 0.0
            qt_new = q_prev

        if B > 0 and qt_new > B:
            qt_new = q0

        Qt[r] = qt_new
        errt[r] = err_t

    dt_Qt = pd.DataFrame({"value": Qt}, index=dates)
    dt_errt = pd.DataFrame({"value": errt}, index=dates)
    dt_infot = pd.DataFrame({"value": infot}, index=dates)

    return dt_Qt, dt_errt, dt_infot
