"""
visualisation.py
================
Generates all plots for the F1 Tyre Degradation ML project.

Charts produced:
    1. Degradation curves per compound (SOFT / MEDIUM / HARD)
    2. Linear Regression vs Random Forest — actual vs predicted
    3. Feature importance (Random Forest)
    4. Model comparison bar chart (MAE and R²)

Usage:
    from src.visualisation import plot_all
    plot_all(results, df_clean)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

from src.models import predict_degradation_curve

# ── Output directory ──────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── F1 colour palette ─────────────────────────────────────────────────────────
COMPOUND_COLOURS = {
    "SOFT":   "#E8002D",   # Pirelli red
    "MEDIUM": "#FFF200",   # Pirelli yellow
    "HARD":   "#FFFFFF",   # Pirelli white
}

BACKGROUND   = "#1a1a2e"   # dark navy — F1 broadcast style
GRID_COLOUR  = "#2a2a4a"
TEXT_COLOUR  = "#e0e0e0"
ACCENT_LR    = "#00d4ff"   # cyan  — Linear Regression
ACCENT_RF    = "#ff6b35"   # orange — Random Forest


def _set_f1_style():
    """Apply F1-broadcast-inspired dark style to all plots."""
    plt.rcParams.update({
        "figure.facecolor":  BACKGROUND,
        "axes.facecolor":    BACKGROUND,
        "axes.edgecolor":    GRID_COLOUR,
        "axes.labelcolor":   TEXT_COLOUR,
        "axes.titlecolor":   TEXT_COLOUR,
        "xtick.color":       TEXT_COLOUR,
        "ytick.color":       TEXT_COLOUR,
        "text.color":        TEXT_COLOUR,
        "grid.color":        GRID_COLOUR,
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "legend.facecolor":  "#2a2a4e",
        "legend.edgecolor":  GRID_COLOUR,
        "legend.labelcolor": TEXT_COLOUR,
    })


# ── Plot 1: Degradation curves per compound ───────────────────────────────────

def plot_degradation_curves(results: dict, max_deg_laps: int = 40):
    """
    Show how predicted lap time rises as tyre age increases,
    separately for SOFT, MEDIUM, and HARD compounds.
    """
    _set_f1_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    compound_map = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
    rf_model     = results["rf"]["model"]
    feature_cols = results["feature_cols"]

    for code, name in compound_map.items():
        curve = predict_degradation_curve(
            model=rf_model,
            compound_code=code,
            deg_laps=max_deg_laps,
            feature_cols=feature_cols,
        )
        ax.plot(
            curve["DegLap"],
            curve["PredictedLapTime"],
            color=COMPOUND_COLOURS[name],
            linewidth=2.5,
            label=name,
        )

    ax.set_xlabel("Tyre Age (laps on current set)")
    ax.set_ylabel("Predicted Lap Time (seconds)")
    ax.set_title("F1 Tyre Degradation Curves by Compound\n(Random Forest — 2023–2025 seasons)")
    ax.legend(title="Compound", loc="upper left")
    ax.grid(True)

    # Annotation explaining what we see
    ax.annotate(
        "Lap time increases as tyre degrades",
        xy=(max_deg_laps * 0.6, ax.get_ylim()[1] * 0.97),
        fontsize=9,
        color=TEXT_COLOUR,
        alpha=0.7,
    )

    fig.tight_layout()
    path = FIGURES_DIR / "degradation_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 2: Actual vs Predicted ───────────────────────────────────────────────

def plot_actual_vs_predicted(results: dict):
    """
    Scatter plot of actual lap times vs model predictions.
    One panel for LR, one for RF — side by side.
    """
    _set_f1_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    y_test = results["y_test"]
    models_data = [
        ("Linear Regression", results["linear"]["predictions"], ACCENT_LR),
        ("Random Forest",     results["rf"]["predictions"],     ACCENT_RF),
    ]

    for ax, (name, preds, colour) in zip(axes, models_data):
        ax.scatter(y_test, preds, alpha=0.3, s=8, color=colour)

        # Perfect prediction line
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color="white", linewidth=1.5, linestyle="--", label="Perfect prediction")

        ax.set_xlabel("Actual Lap Time (s)")
        ax.set_ylabel("Predicted Lap Time (s)")
        ax.set_title(name)
        ax.legend(fontsize=9)
        ax.grid(True)

        # Annotate with R²
        r2_key = "r2"
        r2_val = results["linear"]["r2"] if "Linear" in name else results["rf"]["r2"]
        mae_val = results["linear"]["mae"] if "Linear" in name else results["rf"]["mae"]
        ax.text(
            0.05, 0.92,
            f"R² = {r2_val:.4f}\nMAE = {mae_val:.3f}s",
            transform=ax.transAxes,
            fontsize=10,
            color=colour,
            verticalalignment="top",
        )

    fig.suptitle("Actual vs Predicted Lap Times — Model Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 3: Feature Importance ────────────────────────────────────────────────

def plot_feature_importance(results: dict):
    """
    Horizontal bar chart of Random Forest feature importances.
    Shows which variables matter most for predicting lap time.
    """
    _set_f1_style()
    importance = results["rf"]["feature_importance"].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        importance.index,
        importance.values,
        color=ACCENT_RF,
        edgecolor=BACKGROUND,
        height=0.6,
    )

    # Add value labels on bars
    for bar, val in zip(bars, importance.values):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
            color=TEXT_COLOUR,
        )

    ax.set_xlabel("Feature Importance (Gini impurity reduction)")
    ax.set_title("Random Forest — Feature Importances\nWhat drives F1 lap time prediction?")
    ax.grid(True, axis="x")
    ax.set_xlim(0, importance.values.max() * 1.18)

    fig.tight_layout()
    path = FIGURES_DIR / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Plot 4: Model comparison bar chart ───────────────────────────────────────

def plot_model_comparison(results: dict):
    """
    Side-by-side bar chart comparing Linear Regression vs Random Forest
    on MAE and R² metrics.
    """
    _set_f1_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    models  = ["Linear Regression", "Random Forest"]
    colours = [ACCENT_LR, ACCENT_RF]

    # MAE (lower is better)
    maes = [results["linear"]["mae"], results["rf"]["mae"]]
    axes[0].bar(models, maes, color=colours, edgecolor=BACKGROUND, width=0.5)
    axes[0].set_title("Mean Absolute Error (lower = better)")
    axes[0].set_ylabel("MAE (seconds)")
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.01, f"{v:.3f}s", ha="center", fontsize=11, color=TEXT_COLOUR)
    axes[0].grid(True, axis="y")

    # R² (higher is better)
    r2s = [results["linear"]["r2"], results["rf"]["r2"]]
    axes[1].bar(models, r2s, color=colours, edgecolor=BACKGROUND, width=0.5)
    axes[1].set_title("R² Score (higher = better)")
    axes[1].set_ylabel("R²")
    axes[1].set_ylim(0, 1.05)
    for i, v in enumerate(r2s):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center", fontsize=11, color=TEXT_COLOUR)
    axes[1].grid(True, axis="y")

    fig.suptitle("Linear Regression vs Random Forest — Performance Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIGURES_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BACKGROUND)
    print(f"Saved: {path}")
    plt.close(fig)


# ── Master function ───────────────────────────────────────────────────────────

def plot_all(results: dict, df_clean: pd.DataFrame = None):
    """
    Generate all four plots in one call.

    Parameters
    ----------
    results    : output from train_and_compare()
    df_clean   : cleaned DataFrame (optional, reserved for future plots)
    """
    print("\nGenerating visualisations...\n")
    plot_degradation_curves(results)
    plot_actual_vs_predicted(results)
    plot_feature_importance(results)
    plot_model_comparison(results)
    print(f"\nAll plots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    # Smoke test with synthetic data
    print("Running visualisation smoke test...\n")

    from sklearn.datasets import make_regression
    from src.models import train_and_compare

    np.random.seed(42)
    X_s, y_s = make_regression(n_samples=2000, n_features=8, noise=2.0, random_state=42)
    y_s = (y_s - y_s.min()) / (y_s.max() - y_s.min()) * 20 + 85

    feature_names = [
        "DegLap", "CompoundCode", "LapNumber", "DriverCode",
        "TrackTemp", "AirTemp", "Humidity", "WindSpeed"
    ]
    X_df = pd.DataFrame(X_s, columns=feature_names)
    y_sr = pd.Series(y_s, name="LapTime")

    results = train_and_compare(X_df, y_sr)
    plot_all(results)
    print("\nSmoke test complete — check outputs/figures/")
