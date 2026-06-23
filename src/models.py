"""
models.py
=========
Trains and compares Linear Regression vs Random Forest
for predicting F1 lap time as a function of tyre degradation.

Usage:
    from src.models import train_and_compare
    results = train_and_compare(X, y)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


def train_and_compare(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Train Linear Regression and Random Forest on the same data.
    Compare performance using MAE, R², and cross-validation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix from get_feature_matrix()
    y : pd.Series
        Target lap times in seconds

    Returns
    -------
    dict with keys:
        'linear'  : dict of metrics + fitted model
        'rf'      : dict of metrics + fitted model
        'X_test'  : held-out features (for visualisation)
        'y_test'  : held-out targets
        'scaler'  : fitted scaler (for inverse transforms if needed)
    """

    # ── Train / test split ────────────────────────────────────────────────────
    # 80% train, 20% test — random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # ── Scale features for Linear Regression ─────────────────────────────────
    # LR is sensitive to feature scale; RF is not — so we scale for LR only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 1. Linear Regression ──────────────────────────────────────────────────
    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)

    lr_mae = mean_absolute_error(y_test, lr_preds)
    lr_r2 = r2_score(y_test, lr_preds)

    # Cross-validation (5-fold) on training set
    lr_cv = cross_val_score(
        LinearRegression(), X_train_scaled, y_train,
        cv=5, scoring="r2"
    )

    print(f"  MAE  : {lr_mae:.3f}s")
    print(f"  R²   : {lr_r2:.4f}")
    print(f"  CV R²: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")

    # ── 2. Random Forest ──────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,       # 200 trees — good balance of accuracy vs speed
        max_depth=12,           # prevent overfitting
        min_samples_leaf=5,     # each leaf needs at least 5 laps
        random_state=42,
        n_jobs=-1               # use all CPU cores
    )
    rf.fit(X_train, y_train)    # RF uses raw (unscaled) features
    rf_preds = rf.predict(X_test)

    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    # Cross-validation (5-fold)
    rf_cv = cross_val_score(
        RandomForestRegressor(
            n_estimators=100, max_depth=12,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        X_train, y_train, cv=5, scoring="r2"
    )

    print(f"  MAE  : {rf_mae:.3f}s")
    print(f"  R²   : {rf_r2:.4f}")
    print(f"  CV R²: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    # ── Feature importance (RF only) ──────────────────────────────────────────
    feature_importance = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print(f"\nFeature importances (Random Forest):")
    print(feature_importance.to_string())

    # ── Summary comparison ────────────────────────────────────────────────────
    print("\n" + "="*45)
    print("MODEL COMPARISON SUMMARY")
    print("="*45)
    print(f"{'Metric':<20} {'Linear Reg':>12} {'Random Forest':>14}")
    print("-"*45)
    print(f"{'MAE (seconds)':<20} {lr_mae:>12.3f} {rf_mae:>14.3f}")
    print(f"{'R²':<20} {lr_r2:>12.4f} {rf_r2:>14.4f}")
    print(f"{'CV R² (mean)':<20} {lr_cv.mean():>12.4f} {rf_cv.mean():>14.4f}")
    print(f"{'CV R² (std)':<20} {lr_cv.std():>12.4f} {rf_cv.std():>14.4f}")
    print("="*45)

    winner = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
    print(f"\nBetter model: {winner}")

    return {
        "linear": {
            "model": lr,
            "predictions": lr_preds,
            "mae": lr_mae,
            "r2": lr_r2,
            "cv_r2_mean": lr_cv.mean(),
            "cv_r2_std": lr_cv.std(),
        },
        "rf": {
            "model": rf,
            "predictions": rf_preds,
            "mae": rf_mae,
            "r2": rf_r2,
            "cv_r2_mean": rf_cv.mean(),
            "cv_r2_std": rf_cv.std(),
            "feature_importance": feature_importance,
        },
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "scaler": scaler,
        "feature_cols": list(X.columns),
    }


def predict_degradation_curve(
    model,
    compound_code: int,
    deg_laps: int = 40,
    track_temp: float = 40.0,
    air_temp: float = 28.0,
    driver_code: int = 0,
    lap_number_start: int = 10,
    humidity: float = 50.0,
    wind_speed: float = 5.0,
    scaler=None,
    feature_cols: list = None,
) -> pd.DataFrame:
    """
    Generate a predicted degradation curve for a given compound.

    Holds all variables fixed except DegLap (tyre age), so we can see
    purely how lap time changes as the tyre wears.

    Parameters
    ----------
    model        : fitted sklearn model
    compound_code: 0=SOFT, 1=MEDIUM, 2=HARD
    deg_laps     : how many laps to predict (tyre life range)
    track_temp   : fixed track temperature (°C)
    air_temp     : fixed air temperature (°C)
    driver_code  : encoded driver (0 = reference driver)
    lap_number_start : starting lap number (affects fuel load proxy)
    scaler       : StandardScaler (pass for Linear Regression, None for RF)
    feature_cols : list of feature column names in correct order

    Returns
    -------
    pd.DataFrame with columns: DegLap, PredictedLapTime
    """
    if feature_cols is None:
        feature_cols = [
            "DegLap", "CompoundCode", "LapNumber", "DriverCode",
            "TrackTemp", "AirTemp", "Humidity", "WindSpeed"
        ]

    deg_lap_range = np.arange(1, deg_laps + 1)

    # Build input matrix — vary DegLap, hold everything else fixed
    X_curve = pd.DataFrame({
        "DegLap": deg_lap_range,
        "CompoundCode": compound_code,
        "LapNumber": lap_number_start + deg_lap_range,
        "DriverCode": driver_code,
        "TrackTemp": track_temp,
        "AirTemp": air_temp,
        "Humidity": humidity,
        "WindSpeed": wind_speed,
    })[feature_cols]  # ensure correct column order

    # Scale if Linear Regression
    X_input = scaler.transform(X_curve) if scaler is not None else X_curve

    predictions = model.predict(X_input)

    return pd.DataFrame({
        "DegLap": deg_lap_range,
        "PredictedLapTime": predictions,
        "Compound": {0: "SOFT", 1: "MEDIUM", 2: "HARD"}[compound_code],
    })


if __name__ == "__main__":
    # Smoke test with synthetic data
    print("Running models.py smoke test...\n")

    from sklearn.datasets import make_regression

    np.random.seed(42)
    X_synthetic, y_synthetic = make_regression(
        n_samples=2000, n_features=8,
        noise=2.0, random_state=42
    )
    # Shift y to realistic lap time range (85-105 seconds)
    y_synthetic = (y_synthetic - y_synthetic.min())
    y_synthetic = (y_synthetic / y_synthetic.max()) * 20 + 85

    feature_names = [
        "DegLap", "CompoundCode", "LapNumber", "DriverCode",
        "TrackTemp", "AirTemp", "Humidity", "WindSpeed"
    ]
    X_df = pd.DataFrame(X_synthetic, columns=feature_names)
    y_series = pd.Series(y_synthetic, name="LapTime")

    results = train_and_compare(X_df, y_series)

    # Test degradation curve prediction
    curve = predict_degradation_curve(
        model=results["rf"]["model"],
        compound_code=1,  # MEDIUM
        deg_laps=30,
        feature_cols=results["feature_cols"],
    )
    print(f"\nSample degradation curve (MEDIUM, first 5 laps):")
    print(curve.head())
    print("\nmodels.py smoke test passed.")