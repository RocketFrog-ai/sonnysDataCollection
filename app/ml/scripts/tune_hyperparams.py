"""
Grid search over RF hyperparameters for car wash prediction.
n=531, 50 features, target: cars_washed(Actual). Picks config with best CV correlation, guards against overfitting.
"""
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataSET__1_.xlsx"
TARGET = "cars_washed(Actual)"


def correlation_scorer(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def main():
    df = pd.read_excel(DATA_PATH)
    X = df.drop(columns=[TARGET, "full_site_address"])
    y = df[TARGET]
    for c in X.columns:
        if X[c].dtype in ["float64", "int64"] and X[c].isnull().any():
            X[c] = X[c].fillna(X[c].median())
    assert len(X) == 531 and X.shape[1] == 50

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m, "max_features": f}
        for n in [100, 150, 200, 250, 300]
        for d in [6, 8, 10]
        for m in [10, 15, 20, 25]
        for f in ["sqrt", 0.5]
    ]

    best_corr = -1
    best_params = None
    best_train_corr = None
    results = []

    for i, params in enumerate(param_grid):
        m = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        pred_cv = cross_val_predict(m, X, y, cv=cv)
        val_corr = correlation_scorer(y, pred_cv)
        m.fit(X, y)
        train_pred = m.predict(X)
        train_corr = correlation_scorer(y, train_pred)
        gap = train_corr - val_corr
        results.append((params.copy(), val_corr, train_corr, gap))
        if val_corr > best_corr and gap < 0.25:
            best_corr = val_corr
            best_params = params.copy()
            best_train_corr = train_corr
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(param_grid)} done, best val corr so far: {best_corr:.4f}")

    if best_params is None:
        r = max(results, key=lambda x: x[1])
        best_params, best_corr, best_train_corr = r[0], r[1], r[2]
        print("  (no config with gap < 0.25; picked best CV correlation)")

    print("\n" + "=" * 60)
    print("BEST CONFIG (best CV correlation, overfitting gap < 0.25):")
    print("=" * 60)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  CV correlation:   {best_corr:.4f}")
    print(f"  Train correlation: {best_train_corr:.4f}")

    top5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 by CV correlation:")
    for p, vc, tc, gap in top5:
        print(f"  {vc:.4f}  (train {tc:.4f}, gap {gap:.4f})  {p}")


if __name__ == "__main__":
    main()
