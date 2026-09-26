"""Heterogeneous ensemble for the matcher: LightGBM (existing) + importance-weighted Extra-Trees +
an approximate-RBF kernel model (Nystroem + linear), combined by a stacked logistic regression (the only
place logistic regression is used). Enabled with BER_ENSEMBLE=1 and kept
only if it beats LightGBM alone on out-of-fold macro F0.5 (gate in matching.cmd_train).

Extra-Trees here is a random-subspace ensemble: each member sees a random NUMBER of features, drawn with
probability proportional to the fold's LightGBM gain importance (plus a floor so no feature is never
used), and uses extremely randomized splits. Different algorithm + different feature views = errors that
differ from boosting, which is what makes blending worthwhile."""
import numpy as np
import pandas as pd

ET_MEMBERS = 40
ET_TREES_PER_MEMBER = 5
ET_ROWS = 400_000          # rows per member (random sample of the fold's training rows)
LR_ROWS = 1_000_000


def _xy(X):
    return np.nan_to_num(X.to_numpy(np.float32), nan=-1.0, posinf=1e6, neginf=-1e6)


def fit_extra_trees(X, y, w, importance, seed=0):
    from sklearn.ensemble import ExtraTreesClassifier
    rng = np.random.default_rng(seed)
    cols = list(X.columns)
    imp = np.asarray(importance, dtype=np.float64)
    prob = imp / imp.sum() if imp.sum() > 0 else np.ones(len(cols)) / len(cols)
    prob = 0.9 * prob + 0.1 / len(cols)                    # floor: every feature keeps some chance
    prob /= prob.sum()
    members = []
    n = len(X)
    for i in range(ET_MEMBERS):
        k = int(rng.integers(max(4, len(cols) // 6), max(5, len(cols) // 2) + 1))   # random subset size
        sub = list(rng.choice(len(cols), size=min(k, len(cols)), replace=False, p=prob))
        rows = rng.choice(n, size=min(ET_ROWS, n), replace=False)
        m = ExtraTreesClassifier(n_estimators=ET_TREES_PER_MEMBER, max_depth=18, min_samples_leaf=40,
                                 max_features="sqrt", n_jobs=-1, random_state=int(rng.integers(1 << 30)))
        m.fit(_xy(X.iloc[rows, sub]), y[rows], sample_weight=None if w is None else w[rows])
        members.append(([cols[j] for j in sub], m))
    return members


def predict_extra_trees(members, X):
    p = np.zeros(len(X))
    for cols, m in members:
        p += m.predict_proba(_xy(X[cols]))[:, 1]
    return p / max(len(members), 1)


def fit_rbf(X, y, w, seed=0, rows=300_000, components=400):
    """Approximate-RBF kernel model: impute + scale -> Nystroem RBF features -> linear classifier.
    Smooth, distance-based decision surface: errors unlike both boosting and randomized trees."""
    from sklearn.kernel_approximation import Nystroem
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(rows, len(X)), replace=False)
    Xa = X.iloc[idx]
    miss = Xa.isna().to_numpy(np.float32)
    keep_miss = miss.mean(axis=0) > 0.001
    Z = np.hstack([np.nan_to_num(Xa.to_numpy(np.float64)), miss[:, keep_miss]])
    sc = StandardScaler().fit(Z)
    Zs = np.clip(sc.transform(Z), -6, 6)
    ny = Nystroem(kernel="rbf", gamma=1.0 / Zs.shape[1], n_components=min(components, len(Zs)), random_state=seed).fit(Zs)
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=30, tol=1e-4, random_state=seed)
    clf.fit(ny.transform(Zs), y[idx], sample_weight=None if w is None else w[idx])
    return {"scaler": sc, "nystroem": ny, "model": clf, "keep_miss": keep_miss, "cols": list(X.columns)}


def predict_rbf(m, X, chunk=500_000):
    X = X[m["cols"]]
    out = np.zeros(len(X))
    for i in range(0, len(X), chunk):
        Xc = X.iloc[i:i + chunk]
        miss = Xc.isna().to_numpy(np.float32)
        Z = np.hstack([np.nan_to_num(Xc.to_numpy(np.float64)), miss[:, m["keep_miss"]]])
        out[i:i + chunk] = m["model"].predict_proba(m["nystroem"].transform(np.clip(m["scaler"].transform(Z), -6, 6)))[:, 1]
    return out


# kept names used by matching.py: the third base model is now the approximate-RBF model
fit_logreg, predict_logreg = fit_rbf, predict_rbf


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stack_matrix(p_lgb, p_et, p_lr, srcn, a_missing):
    return np.column_stack([_logit(p_lgb), _logit(p_et), _logit(p_lr), (np.asarray(srcn) == 3).astype(float),
                            (np.asarray(a_missing) > 0).astype(float)])


def fit_stacker(Z, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=300).fit(Z, y)


def stacked_oof(Z, y, fold):
    """Out-of-fold stacked probabilities (stacker trained on the other folds' base predictions)."""
    out = np.zeros(len(y))
    for k in np.unique(fold):
        te = fold == k
        out[te] = fit_stacker(Z[~te], y[~te]).predict_proba(Z[te])[:, 1]
    return out


def predict_ensemble(ens, X, p_lgb, srcn, a_missing):
    p_et = np.mean([predict_extra_trees(m, X) for m in ens["et"]], axis=0)
    p_lr = np.mean([predict_rbf(m, X) for m in ens["lr"]], axis=0)
    return ens["stacker"].predict_proba(stack_matrix(p_lgb, p_et, p_lr, srcn, a_missing))[:, 1]
