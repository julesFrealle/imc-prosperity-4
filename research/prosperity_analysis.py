"""
prosperity_analysis.py
======================
IMC Prosperity — clean analysis toolkit.

Drop this file next to your data folder and import it in your notebook.

File naming conventions expected
---------------------------------
Current year  : prices_round_0_day_-2.csv   / trades_round_0_day_-2.csv
Previous years: prices_round_1_day_0_P1.csv / trades_round_1_day_0_P1.csv
                (suffix _P1, _P2, _P3 ...)

As soon as you drop a new file into the folder the loader picks it up
automatically — no manual list to update.
"""

from __future__ import annotations

import os
import re
import glob
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False
    warnings.warn(
        "plotly not installed — all charts will use matplotlib. "
        "Run `pip install plotly` for interactive charts.",
        ImportWarning,
        stacklevel=2,
    )

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

PRICE_COLS = [
    "day", "timestamp", "product",
    "bid_price_1", "bid_volume_1", "bid_price_2", "bid_volume_2",
    "bid_price_3", "bid_volume_3",
    "ask_price_1", "ask_volume_1", "ask_price_2", "ask_volume_2",
    "ask_price_3", "ask_volume_3",
    "mid_price", "profit_and_loss",
]

TRADE_COLS = ["timestamp", "buyer", "seller", "symbol", "currency", "price", "quantity"]

TIMESTAMP_STRIDE = 1_000_000   # each day is 1 000 000 ticks apart

# colour palette for edition overlays
EDITION_COLORS = {
    "Current": "#1f77b4",
    "P1":      "#ff7f0e",
    "P2":      "#2ca02c",
    "P3":      "#d62728",
    "P4":      "#9467bd",
}


# ──────────────────────────────────────────────────────────────────────────────
# 1 · DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def _read_csv_robust(path: str) -> pd.DataFrame:
    from io import StringIO

    encodings = ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin1", "cp1252")

    last_error = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="replace") as f:
                text = f.read()

            text = text.replace("\x00", "")

            return pd.read_csv(
                StringIO(text),
                sep=";",
                engine="python",
                header=0
            )

        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read CSV file: {path}. Last error: {last_error}")

def _parse_day_number(filename: str) -> float:
    """Extract the day number from a filename like prices_round_0_day_-2.csv → -2."""
    m = re.search(r"day_(-?\d+)", os.path.basename(filename))
    return float(m.group(1)) if m else 0.0


def _read_price_csv(path: str, global_offset: int) -> pd.DataFrame:
    """Read one semicolon-separated prices CSV and apply a global timestamp offset."""
    df = _read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]

    # If still one column, split manually
    if df.shape[1] == 1:
        df = df.iloc[:, 0].astype(str).str.split(";", expand=True)

    # Keep only expected number of columns
    df = df.iloc[:, :len(PRICE_COLS)]
    df.columns = PRICE_COLS[:df.shape[1]]

    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column found in {path}. Columns found: {df.columns.tolist()}")

    # Drop repeated headers / empty rows
    df = df[df["timestamp"].notna()]
    df = df[df["timestamp"] != "timestamp"]

    numeric = [c for c in df.columns if c not in ("day", "product")]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")

    bad_ts = df["timestamp"].isna().sum()
    if bad_ts:
        print(f"Warning: dropping {bad_ts} bad timestamp rows from {path}")
        df = df[df["timestamp"].notna()].copy()

    df["timestamp"] = df["timestamp"].astype(int) + global_offset
    return df


def _read_trade_csv(path: str, global_offset: int) -> pd.DataFrame:
    """Read one semicolon-separated trades CSV and apply a global timestamp offset."""
    df = _read_csv_robust(path)
    df.columns = [str(c).strip() for c in df.columns]

    if df.shape[1] == 1:
        df = df.iloc[:, 0].astype(str).str.split(";", expand=True)

    df = df.iloc[:, :len(TRADE_COLS)]
    df.columns = TRADE_COLS[:df.shape[1]]

    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column found in {path}. Columns found: {df.columns.tolist()}")

    df = df[df["timestamp"].notna()]
    df = df[df["timestamp"] != "timestamp"]

    numeric = ["timestamp", "price", "quantity"]
    existing_numeric = [c for c in numeric if c in df.columns]
    df[existing_numeric] = df[existing_numeric].apply(pd.to_numeric, errors="coerce")

    bad_ts = df["timestamp"].isna().sum()
    if bad_ts:
        print(f"Warning: dropping {bad_ts} bad timestamp rows from {path}")
        df = df[df["timestamp"].notna()].copy()

    df["timestamp"] = df["timestamp"].astype(int) + global_offset
    return df


def _collect_files(kind: str, edition: Optional[str], base: str = "Prices") -> list[str]:
    """
    Resolve the right folder and glob for price/trade files.

    Folder layout expected (matches your VS Code project):
        Prices/                ← current year
        Prices/P1/             ← Prosperity 1
        Prices/P2/             ← Prosperity 2
        Prices/P3/             ← Prosperity 3

    kind    : "prices" or "trades"
    edition : None → current year  |  "P1" | "P2" | "P3" → sub-folder
    base    : root prices folder (default "Prices")
    """
    folder = os.path.join(base, edition) if edition else base
    pattern = os.path.join(folder, f"{kind}_round_*_day_*.csv")
    files   = glob.glob(pattern)
    if not files:
        # Fallback: maybe the user kept the _P1 suffix style in the same folder
        pattern2 = os.path.join(base, f"{kind}_round_*_day_*_{edition}.csv") if edition else pattern
        files    = glob.glob(pattern2)
    return sorted(files, key=_parse_day_number)


def load_prices(base: str = "Prices", edition: Optional[str] = None) -> pd.DataFrame:
    """
    Auto-detect and load all price files.

    Parameters
    ----------
    base    : root prices folder — "Prices" by default (matches your project layout)
    edition : None → Prices/          (current year)
              "P1" → Prices/P1/
              "P2" → Prices/P2/
              "P3" → Prices/P3/

    Returns
    -------
    pd.DataFrame with all days concatenated. Each file gets a timestamp offset
    of index × 1_000_000 so days never overlap on the time axis.
    """
    files = _collect_files("prices", edition, base)
    if not files:
        folder = os.path.join(base, edition) if edition else base
        raise FileNotFoundError(
            f"No price files found in '{folder}'.\n"
            f"Expected files matching: prices_round_*_day_*.csv"
        )
    dfs = []
    for i, path in enumerate(files):
        dfs.append(_read_price_csv(path, global_offset=i * TIMESTAMP_STRIDE))
    df = pd.concat(dfs, ignore_index=True)
    df["_file_idx"] = df["timestamp"] // TIMESTAMP_STRIDE
    return df


def load_trades(base: str = "Prices", edition: Optional[str] = None) -> pd.DataFrame:
    """
    Auto-detect and load all trade files.

    Parameters
    ----------
    base    : root prices folder — "Prices" by default
    edition : None → Prices/  |  "P1" -> Prices/P1/  |  "P2" -> Prices/P2/ ...

    Returns an empty DataFrame (with correct columns) if no files are found,
    so the notebook keeps running even when trade files are missing.
    """
    files = _collect_files("trades", edition, base)
    if not files:
        folder = os.path.join(base, edition) if edition else base
        warnings.warn(
            f"No trade files found in '{folder}' — returning empty DataFrame.\n"
            f"Expected: trades_round_*_day_*.csv\n"
            f"Run debug_files(\"{base}\") to see what is in the folder.",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(columns=TRADE_COLS)
    dfs = []
    for i, path in enumerate(files):
        dfs.append(_read_trade_csv(path, global_offset=i * TIMESTAMP_STRIDE))
    return pd.concat(dfs, ignore_index=True)


def debug_files(base: str = "Prices") -> None:
    """
    Print every CSV found under `base` and its sub-folders.
    Run this if a loader fails to find files.
    """
    print(f"Scanning \"{base}\" ...")
    found = False
    for root, dirs, files in os.walk(base):
        dirs.sort()
        for fname in sorted(files):
            if fname.endswith(".csv"):
                print(f"  {os.path.join(root, fname)}")
                found = True
    if not found:
        print(f"  No CSV files found under \"{base}\".")
        print(f"  Working directory : {os.getcwd()}")
        print(f"  Contents of cwd   : {sorted(os.listdir('.'))}")


def get_products(df: pd.DataFrame, col: str = "product") -> list[str]:
    """Return a sorted list of unique product names in the DataFrame."""
    col = col if col in df.columns else "symbol"
    return sorted(df[col].dropna().unique().tolist())


# ──────────────────────────────────────────────────────────────────────────────
# 2 · FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def enrich(
    df: pd.DataFrame,
    product: str,
    window: int = 100,
    smooth_window: int = 3,
) -> pd.DataFrame:
    """
    Filter to a single product and compute analysis columns:
      - mid_price      : (best_ask + best_bid) / 2
      - spread_ba      : best_ask − best_bid  (bid-ask spread)
      - ma             : rolling mean of mid_price
      - std            : rolling std of mid_price
      - z_score        : (mid_price − ma) / std
      - relative_diff  : 100 × (mid_price − ma) / ma
      - z_velocity     : first difference of z_score
      - z_velocity_s   : smoothed z_velocity (smooth_window-period MA)

    Parameters
    ----------
    df      : full prices DataFrame (all products)
    product : product name to filter
    window        : rolling window size
    smooth_window : smoothing window for z-score velocity
    """
    p = df[df["product"] == product].copy().reset_index(drop=True)
    if p.empty:
        raise ValueError(f"Product '{product}' not found. Available: {get_products(df)}")

    p["mid_price"]     = (p["ask_price_1"] + p["bid_price_1"]) / 2
    p["spread_ba"]     = p["ask_price_1"] - p["bid_price_1"]
    p["ma"]            = p["mid_price"].rolling(window).mean()
    p["std"]           = p["mid_price"].rolling(window).std()
    p["z_score"]       = (p["mid_price"] - p["ma"]) / p["std"]
    p["relative_diff"] = 100 * (p["mid_price"] - p["ma"]) / p["ma"]
    if smooth_window < 1:
        raise ValueError("smooth_window must be >= 1")

    p["z_velocity"]    = p["z_score"].diff()
    p["z_velocity_s"]  = p["z_velocity"].rolling(smooth_window).mean()
    return p


def compute_spread(df: pd.DataFrame, formula: dict[str, float]) -> pd.DataFrame:
    """
    Compute a custom spread as a linear combination of product mid-prices.

    Parameters
    ----------
    df      : full prices DataFrame
    formula : dict mapping product name → coefficient
              e.g. {"SQUID_INK": 1, "KELP": -1}
                   {"PICNIC_BASKET1": 1, "CROISSANTS": -6, "JAMS": -3, "DJEMBES": -1}

    Returns
    -------
    DataFrame with columns: timestamp, <product>_mid, ..., spread
    """
    pieces = {}
    for product, coeff in formula.items():
        sub = df[df["product"] == product][["timestamp", "ask_price_1", "bid_price_1"]].copy()
        sub[f"{product}_mid"] = (sub["ask_price_1"] + sub["bid_price_1"]) / 2
        pieces[product] = sub[["timestamp", f"{product}_mid"]]

    merged = None
    for product, sub in pieces.items():
        merged = sub if merged is None else merged.merge(sub, on="timestamp", how="inner")

    merged = merged.dropna().reset_index(drop=True)
    merged["spread"] = sum(
        coeff * merged[f"{product}_mid"] for product, coeff in formula.items()
    )
    return merged


def enrich_spread(merged: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    """Add rolling MA, std, and z-score columns to a spread DataFrame."""
    merged = merged.copy()
    merged["spread_ma"]  = merged["spread"].rolling(window).mean()
    merged["spread_std"] = merged["spread"].rolling(window).std()
    merged["z_score"]    = (merged["spread"] - merged["spread_ma"]) / merged["spread_std"]
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# 3 · SINGLE-PRODUCT PLOTS
# ──────────────────────────────────────────────────────────────────────────────

def plot_price_evolution(
    df: pd.DataFrame,
    product: str,
    window: int = 100,
    z_thresh: float = 3.7,
    interactive: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Mid-price + rolling MA on primary axis, Z-score on secondary axis.

    Parameters
    ----------
    df          : full prices DataFrame
    product     : product to plot
    window      : rolling window
    z_thresh    : horizontal dashed line on z-score axis
    interactive : use Plotly (True) or matplotlib (False)
    title       : custom plot title
    """
    p = enrich(df, product, window, smooth_window=3)
    t = title or f"{product} — Mid Price & Z-Score (window={window})"

    if interactive and PLOTLY:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(
            x=p["timestamp"], y=p["mid_price"], name="Mid Price",
            line=dict(color="#1f77b4", width=2),
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=p["timestamp"], y=p["ma"], name=f"MA({window})",
            line=dict(color="darkorange", width=1.5, dash="dash"),
        ), secondary_y=False)

        fig.add_trace(go.Scatter(
            x=p["timestamp"], y=p["z_score"], name="Z-Score",
            line=dict(color="purple", width=1.5),
            opacity=0.4,
        ), secondary_y=True)

        for th, col in [(z_thresh, "red"), (-z_thresh, "red"), (0, "gray")]:
            fig.add_hline(y=th, line_dash="dash", line_color=col,
                          opacity=0.5, secondary_y=True)

        fig.update_layout(
            title=t, template="plotly_white",
            height=600, legend=dict(orientation="h", y=1.05),
            xaxis=dict(domain=[0, 0.94]),
        )
        fig.update_yaxes(title_text="Mid Price", secondary_y=False)
        fig.update_yaxes(title_text="Z-Score",   secondary_y=True)
        fig.show()

    else:
        fig, ax1 = plt.subplots(figsize=(22, 8))
        ax2 = ax1.twinx()

        ax1.plot(p["timestamp"], p["mid_price"], color="#1f77b4", linewidth=2, label="Mid Price", zorder=2)
        ax1.plot(p["timestamp"], p["ma"],        color="darkorange", linewidth=1.5,
                 linestyle="--", label=f"MA({window})", zorder=2)
        ax1.set_ylabel("Mid Price", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        ax2.plot(p["timestamp"], p["z_score"], color="purple", linewidth=1,
                 linestyle="--", alpha=0.5, label="Z-Score", zorder=1)
        ax2.axhline( z_thresh, linestyle="--", color="red",  alpha=0.5)
        ax2.axhline(-z_thresh, linestyle="--", color="red",  alpha=0.5)
        ax2.axhline(0,          linestyle="--", color="gray", alpha=0.4)
        ax2.set_ylabel("Z-Score", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=10)
        plt.title(t, fontsize=14)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_bid_ask_spread(
    df: pd.DataFrame,
    product: str,
    window: int = 100,
    interactive: bool = True,
) -> None:
    """
    Plot the bid-ask spread (ask_price_1 − bid_price_1) over time
    with a rolling mean overlay.
    """
    p = enrich(df, product, window)
    ba_ma = p["spread_ba"].rolling(window).mean()
    t = f"{product} — Bid-Ask Spread"

    if interactive and PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["spread_ba"],
                                 name="Bid-Ask Spread", line=dict(color="#1f77b4"), opacity=0.6))
        fig.add_trace(go.Scatter(x=p["timestamp"], y=ba_ma,
                                 name=f"MA({window})", line=dict(color="darkorange", dash="dash")))
        fig.update_layout(title=t, template="plotly_white",
                          height=400, xaxis_title="Timestamp", yaxis_title="Spread")
        fig.show()
    else:
        plt.figure(figsize=(22, 5))
        plt.plot(p["timestamp"], p["spread_ba"], alpha=0.6, label="Bid-Ask Spread", color="#1f77b4")
        plt.plot(p["timestamp"], ba_ma, linestyle="--", color="darkorange", label=f"MA({window})")
        plt.title(t)
        plt.xlabel("Timestamp")
        plt.ylabel("Spread")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_zscore_velocity(
    df: pd.DataFrame,
    product: str,
    window: int = 100,
    smooth_window: int = 3,
    interactive: bool = True,
) -> None:
    """
    Mid Price + MA on primary axis, Z-Score velocity (diff + smoothed) on secondary.
    Useful to spot momentum reversals.
    """
    p = enrich(df, product, window, smooth_window=smooth_window)
    t = f"{product} — Mid Price & Z-Score Velocity"

    if interactive and PLOTLY:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["mid_price"], name="Mid Price",
                                 line=dict(color="#1f77b4", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["ma"], name=f"MA({window})",
                                 line=dict(color="darkorange", dash="dash")), secondary_y=False)
        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["z_velocity"], name="Z Velocity (raw)",
                                 line=dict(color="purple", width=1, dash="dot"),
                                 opacity=0.3), secondary_y=True)
        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["z_velocity_s"],
                                 name=f"Z Velocity (smooth {smooth_window})",
                                 line=dict(color="green", width=1.5),
                                 opacity=0.8), secondary_y=True)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, secondary_y=True)

        fig.update_layout(title=t, template="plotly_white", height=600,
                          xaxis=dict(domain=[0, 0.94]),
                          legend=dict(orientation="h", y=1.05))
        fig.update_yaxes(
        title=dict(text="Mid Price", font=dict(color="#1f77b4")),
        secondary_y=False
        )

        fig.update_yaxes(
        title=dict(text="Z Velocity", font=dict(color="green")),
        secondary_y=True
        )
        fig.show()
    else:
        fig, ax1 = plt.subplots(figsize=(22, 8))
        ax2 = ax1.twinx()
        ax1.plot(p["timestamp"], p["mid_price"], color="#1f77b4",  linewidth=2, label="Mid Price")
        ax1.plot(p["timestamp"], p["ma"],        color="darkorange", linewidth=1.5,
                 linestyle="--", label=f"MA({window})")
        ax1.set_ylabel("Mid Price", color="#1f77b4")

        ax2.plot(p["timestamp"], p["z_velocity"],   color="purple", linewidth=1,
                 linestyle=":",  alpha=0.3, label="Z Velocity")
        ax2.plot(p["timestamp"], p["z_velocity_s"], color="green",  linewidth=1.5,
                 alpha=0.8, label=f"Z Velocity smooth({smooth_window})")
        ax2.axhline(0, linestyle="--", color="gray", alpha=0.4)
        ax2.set_ylabel("Z Velocity", color="green")

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left")
        plt.title(t)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 4 · MULTI-PRODUCT SPREAD ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def _formula_label(formula: dict[str, float]) -> str:
    """Human-readable label for a spread formula."""
    parts = []
    for prod, coeff in formula.items():
        if coeff == 1:
            parts.append(prod)
        elif coeff == -1:
            parts.append(f"−{prod}")
        elif coeff > 0:
            parts.append(f"{coeff}×{prod}")
        else:
            parts.append(f"{coeff}×{prod}")
    return " + ".join(parts).replace("+ −", "− ")


def plot_spread(
    df: pd.DataFrame,
    formula: dict[str, float],
    window: int = 50,
    interactive: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Plot the spread defined by `formula` with rolling MA and ±2 STD bands.

    Example
    -------
    plot_spread(df, {"TOMATOES": 1, "EMERALDS": -1})
    plot_spread(df, {"PICNIC_BASKET1": 1, "CROISSANTS": -6, "JAMS": -3, "DJEMBES": -1})
    """
    m = enrich_spread(compute_spread(df, formula), window)
    t = title or f"Spread: {_formula_label(formula)}  (window={window})"
    mean_val = m["spread"].mean()

    if interactive and PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["spread"],
                                 name="Spread", line=dict(color="darkorange")))
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["spread_ma"],
                                 name=f"MA({window})", line=dict(color="#1f77b4")))
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["spread_ma"] + 2*m["spread_std"],
                                 name="+2 STD", line=dict(dash="dash", color="green")))
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["spread_ma"] - 2*m["spread_std"],
                                 name="−2 STD", line=dict(dash="dash", color="red")))
        fig.add_hline(y=mean_val, line_dash="dot", line_color="gray",
                      annotation_text=f"mean={mean_val:.2f}")
        fig.update_layout(title=t, template="plotly_white", height=500,
                          xaxis_title="Timestamp", yaxis_title="Spread")
        fig.show()
    else:
        plt.figure(figsize=(22, 7))
        plt.plot(m["timestamp"], m["spread"],                        color="darkorange", label="Spread")
        plt.plot(m["timestamp"], m["spread_ma"],                     color="#1f77b4",   label=f"MA({window})")
        plt.plot(m["timestamp"], m["spread_ma"] + 2*m["spread_std"], color="green",     linestyle="--", label="+2 STD")
        plt.plot(m["timestamp"], m["spread_ma"] - 2*m["spread_std"], color="red",       linestyle="--", label="−2 STD")
        plt.axhline(mean_val, linestyle=":", color="gray", label=f"mean={mean_val:.2f}")
        plt.title(t)
        plt.xlabel("Timestamp")
        plt.ylabel("Spread")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_spread_zscore(
    df: pd.DataFrame,
    formula: dict[str, float],
    window: int = 50,
    z_thresh: float = 2.0,
    interactive: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Two-panel interactive plot: spread + MA (top), Z-score (bottom).
    Or single-panel with dual axis if interactive.

    Parameters
    ----------
    z_thresh : horizontal threshold lines on the Z-score panel
    """
    m = enrich_spread(compute_spread(df, formula), window)
    t = title or f"Spread Z-Score: {_formula_label(formula)}  (window={window})"

    if interactive and PLOTLY:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=("Spread + MA", "Z-Score"),
        )
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["spread"],
                                 name="Spread", line=dict(color="darkorange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["spread_ma"],
                                 name=f"MA({window})", line=dict(color="#1f77b4")), row=1, col=1)
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["spread_ma"] + 2*m["spread_std"],
                                 name="+2 STD", line=dict(dash="dash", color="green")), row=1, col=1)
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["spread_ma"] - 2*m["spread_std"],
                                 name="−2 STD", line=dict(dash="dash", color="red")), row=1, col=1)

        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["z_score"],
                                 name="Z-Score", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y= z_thresh, line_dash="dash", line_color="green",  row=2, col=1)
        fig.add_hline(y=-z_thresh, line_dash="dash", line_color="red",    row=2, col=1)
        fig.add_hline(y=0,          line_dash="dot",  line_color="gray",   row=2, col=1)

        fig.update_layout(title=t, template="plotly_white",
                          height=800, showlegend=True)
        fig.update_yaxes(title_text="Spread",  row=1, col=1)
        fig.update_yaxes(title_text="Z-Score", row=2, col=1)
        fig.show()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 12), sharex=True)

        ax1.plot(m["timestamp"], m["spread"],                        color="darkorange", label="Spread")
        ax1.plot(m["timestamp"], m["spread_ma"],                     color="#1f77b4",   label=f"MA({window})", linestyle="--")
        ax1.plot(m["timestamp"], m["spread_ma"] + 2*m["spread_std"], color="green",     linestyle="--", alpha=0.6, label="+2 STD")
        ax1.plot(m["timestamp"], m["spread_ma"] - 2*m["spread_std"], color="red",       linestyle="--", alpha=0.6, label="−2 STD")
        ax1.set_ylabel("Spread")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(t)

        ax2.plot(m["timestamp"], m["z_score"], color="purple", label="Z-Score")
        ax2.axhline( z_thresh, linestyle="--", color="green", alpha=0.6)
        ax2.axhline(-z_thresh, linestyle="--", color="red",   alpha=0.6)
        ax2.axhline(0,          linestyle=":",  color="gray",  alpha=0.5)
        ax2.set_ylabel("Z-Score")
        ax2.set_xlabel("Timestamp")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def spread_stats(
    df: pd.DataFrame,
    formula: dict[str, float],
    checkpoints: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Print and return a summary table of spread statistics.

    Parameters
    ----------
    df          : full prices DataFrame
    formula     : spread formula (same as plot_spread)
    checkpoints : list of timestamps at which to compute cumulative mean/std.
                  If None, uses [1M, 2M, 3M, 4M, 5M, 6M, 7M].

    Returns
    -------
    pd.DataFrame with rows per checkpoint
    """
    m = compute_spread(df, formula)
    if checkpoints is None:
        max_ts = int(m["timestamp"].max())
        checkpoints = list(range(TIMESTAMP_STRIDE, max_ts + TIMESTAMP_STRIDE, TIMESTAMP_STRIDE))

    overall_mean = m["spread"].mean()
    overall_std  = m["spread"].std()
    print(f"Spread formula  : {_formula_label(formula)}")
    print(f"Overall  mean   : {overall_mean:.4f}")
    print(f"Overall  std    : {overall_std:.4f}")
    print()

    rows = []
    for ts in checkpoints:
        snap = m[m["timestamp"] <= ts]["spread"]
        rows.append({
            "timestamp": ts,
            "cumul_mean": round(snap.mean(), 4) if len(snap) else None,
            "cumul_std":  round(snap.std(),  4) if len(snap) else None,
            "n_obs":      len(snap),
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5 · MULTI-YEAR / MULTI-EDITION COMPARISON
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_zscore(series: pd.Series) -> pd.Series:
    """Standardise a series to zero mean, unit std — makes products comparable."""
    mu, sigma = series.mean(), series.std()
    return (series - mu) / sigma if sigma > 0 else series - mu


def match_product_across_editions(
    current_product: str,
    df_current: pd.DataFrame,
    edition_dfs: dict[str, pd.DataFrame],
    window: int = 100,
    top_n: int = 5,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Find which products from previous editions look most like `current_product`
    this year, ranked by similarity metrics.

    The comparison is done on the **normalised Z-score series** of each product
    (zero mean, unit std, same rolling window) so that products with very
    different price levels can still be compared on shape/dynamics.

    Metrics returned
    ----------------
    - pearson_r  : Pearson correlation of Z-score series  (−1 → +1, higher = more similar)
    - r2         : R² of a simple OLS fit  (0 → 1, higher = better linear fit)
    - rmse       : RMSE between Z-score series  (lower = more similar)
    - dtw        : Dynamic Time Warping distance if `dtaidistance` is installed,
                   else NaN  (lower = more similar)

    Parameters
    ----------
    current_product : product in df_current to use as the reference
    df_current      : current year prices DataFrame
    edition_dfs     : dict of {"P1": df_p1, "P2": df_p2, ...}
    window          : rolling window for Z-score computation
    top_n           : how many best matches to print and plot
    plot            : whether to draw the overlay chart

    Returns
    -------
    pd.DataFrame sorted by pearson_r descending, one row per (edition, product)
    """
    # External dependencies used in this file:
    # - numpy
    # - pandas
    # - matplotlib (and matplotlib.gridspec)
    # - plotly (optional)
    # - scikit-learn (sklearn)
    # - dtaidistance (optional for DTW; see usage below)
    # Make sure to have these installed:
    # pip install numpy pandas matplotlib plotly scikit-learn dtaidistance

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # ── Reference series ──────────────────────────────────────────────────────
    ref = enrich(df_current, current_product, window)
    ref_z = _normalise_zscore(ref["mid_price"].dropna()).reset_index(drop=True)
    n_ref = len(ref_z)

    rows = []
    for edition_label, df_ed in edition_dfs.items():
        for product in get_products(df_ed):
            try:
                cand = enrich(df_ed, product, window)
                cand_z = _normalise_zscore(cand["mid_price"].dropna()).reset_index(drop=True)

                # Align lengths — trim to the shorter series
                n = min(n_ref, len(cand_z))
                if n < window * 2:
                    continue   # too short to be meaningful
                a, b = ref_z.iloc[:n].values, cand_z.iloc[:n].values

                # Pearson r
                pearson_r = float(np.corrcoef(a, b)[0, 1])

                # R²
                r2 = float(r2_score(a, b))

                # RMSE
                rmse = float(np.sqrt(np.mean((a - b) ** 2)))

                rows.append({
                    "edition":   edition_label,
                    "product":   product,
                    "pearson_r": round(pearson_r, 4),
                    "r2":        round(r2, 4),
                    "rmse":      round(rmse, 4),
                    "n_obs":     n,
                    "_cand_z":   cand_z,   # kept for plotting, dropped before return
                })
            except Exception as exc:
                print(f"  ⚠  Skipping {edition_label}/{product}: {exc}")

    if not rows:
        print("No candidates found.")
        return pd.DataFrame()

    result = pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Reference : {current_product} (current year, {n_ref} obs)")
    print(f"{'Rank':<5} {'Edition':<8} {'Product':<30} {'Pearson r':>10} {'R²':>8} {'RMSE':>8} {'N':>7}")
    print("─" * 75)
    for rank, row in result.head(top_n).iterrows():
        ed, prod = str(row["edition"]), str(row["product"])
        print(f"{rank+1:<5} {ed:<8} {prod:<30} "
              f"{row['pearson_r']:>10.4f} {row['r2']:>8.4f} {row['rmse']:>8.4f} {int(row['n_obs']):>7,}")
    # ── Plot top N overlays ───────────────────────────────────────────────────
    if plot:
        top_rows = result.head(top_n)
        colors = plt.cm.tab10(np.linspace(0, 0.9, top_n))

        plt.figure(figsize=(22, 7))
        plt.plot(ref_z.values, color="black", linewidth=2.5,
                 label=f"[CURRENT] {current_product}", zorder=10)

        for i, (_, row) in enumerate(top_rows.iterrows()):
            n = min(n_ref, len(row["_cand_z"]))
            plt.plot(row["_cand_z"].iloc[:n].values,
                     color=colors[i], linewidth=1.5, alpha=0.75,
                     label=f"[{str(row['edition'])}] {str(row['product'])}  r={row['pearson_r']:.3f}")

        plt.title(f"Similarity: {current_product} (current) vs top {top_n} matches from previous editions",
                  fontsize=13)
        plt.xlabel("Time steps (aligned)")
        plt.ylabel("Normalised Z-score")
        plt.legend(fontsize=9, loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Drop the raw series column before returning
    result = result.drop(columns=["_cand_z"])
    return result


def compare_editions(
    dfs: dict[str, pd.DataFrame],
    product: str,
    window: int = 100,
    normalise: bool = True,
    interactive: bool = True,
) -> None:
    """
    Overlay the mid-price of the SAME product across multiple editions.

    Parameters
    ----------
    dfs       : {"Current": df, "P1": df_p1, ...}
    product   : must exist in at least one DataFrame
    normalise : subtract the first valid MA so all curves start near 0
    """
    t = f"{product} — Price Evolution Comparison"

    if interactive and PLOTLY:
        fig = go.Figure()
        for label, df in dfs.items():
            if product not in get_products(df):
                print(f"  ⚠  '{product}' not found in edition '{label}', skipping.")
                continue
            p = enrich(df, product, window)
            series = p["mid_price"] - p["ma"].iloc[window] if normalise else p["mid_price"]
            color  = EDITION_COLORS.get(label, None)
            fig.add_trace(go.Scatter(
                x=p["timestamp"], y=series, name=label,
                line=dict(color=color, width=2),
            ))
        fig.update_layout(
            title=t + (" (normalised to MA at t=window)" if normalise else ""),
            template="plotly_white", height=550,
            xaxis_title="Timestamp (relative)",
            yaxis_title="Mid Price" + (" − MA0" if normalise else ""),
            legend=dict(orientation="h", y=1.05),
        )
        fig.show()
    else:
        plt.figure(figsize=(22, 8))
        for label, df in dfs.items():
            if product not in get_products(df):
                continue
            p = enrich(df, product, window)
            series = p["mid_price"] - p["ma"].iloc[window] if normalise else p["mid_price"]
            color  = EDITION_COLORS.get(label, None)
            plt.plot(p["timestamp"], series, label=label, color=color, linewidth=2)
        plt.title(t)
        plt.xlabel("Timestamp (relative)")
        plt.ylabel("Mid Price" + (" − MA0" if normalise else ""))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def compare_spread_editions(
    dfs: dict[str, pd.DataFrame],
    formula: dict[str, float],
    window: int = 50,
    interactive: bool = True,
) -> None:
    """
    Overlay the spread Z-score for the same formula across multiple editions.
    """
    t = f"Spread Z-Score Comparison: {_formula_label(formula)}"

    if interactive and PLOTLY:
        fig = go.Figure()
        for label, df in dfs.items():
            try:
                m = enrich_spread(compute_spread(df, formula), window)
                color = EDITION_COLORS.get(label, None)
                fig.add_trace(go.Scatter(
                    x=m["timestamp"], y=m["z_score"], name=label,
                    line=dict(color=color, width=2),
                ))
            except Exception as exc:
                print(f"  ⚠  Skipping '{label}': {exc}")
        fig.add_hline(y= 2, line_dash="dash", line_color="green", opacity=0.5)
        fig.add_hline(y=-2, line_dash="dash", line_color="red",   opacity=0.5)
        fig.add_hline(y= 0, line_dash="dot",  line_color="gray",  opacity=0.4)
        fig.update_layout(title=t, template="plotly_white", height=500,
                          xaxis_title="Timestamp (relative)", yaxis_title="Z-Score",
                          legend=dict(orientation="h", y=1.05))
        fig.show()
    else:
        plt.figure(figsize=(22, 7))
        for label, df in dfs.items():
            try:
                m = enrich_spread(compute_spread(df, formula), window)
                color = EDITION_COLORS.get(label, None)
                plt.plot(m["timestamp"], m["z_score"], label=label, color=color, linewidth=2)
            except Exception as exc:
                print(f"  ⚠  Skipping '{label}': {exc}")
        plt.axhline( 2, linestyle="--", color="green", alpha=0.5)
        plt.axhline(-2, linestyle="--", color="red",   alpha=0.5)
        plt.axhline( 0, linestyle=":",  color="gray",  alpha=0.4)
        plt.title(t)
        plt.xlabel("Timestamp (relative)")
        plt.ylabel("Z-Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# 6 · QUICK OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────


def halfday_fair_values(
    df: pd.DataFrame,
    product: Optional[str] = None,
    digits: int = 2,
) -> pd.DataFrame:
    """
    Compute evolving fair-value tables by day and by intra-day period.

    For each product/day, this reports the mean of:
      - mid price
      - best bid
      - best ask

    Periods reported:
      - full_day
      - first_half  : local timestamp inside the day < 500_000
      - second_half : local timestamp inside the day >= 500_000

    Parameters
    ----------
    df      : full prices DataFrame, or one-product slice
    product : optional product name to filter
    digits  : rounding used in the returned table

    Returns
    -------
    DataFrame with one row per day/period:
        day | period | fair_mid | fair_best_bid | fair_best_ask
    """
    columns = ["day", "period", "fair_mid", "fair_best_bid", "fair_best_ask"]

    if df.empty:
        return pd.DataFrame(columns=columns)

    required = {"timestamp", "bid_price_1", "ask_price_1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "halfday_fair_values requires these columns: "
            f"{sorted(required)}. Missing: {sorted(missing)}"
        )

    d = df.copy()

    if product is not None:
        product_col = "product" if "product" in d.columns else "symbol"
        d = d[d[product_col] == product].copy()

    if d.empty:
        return pd.DataFrame(columns=columns)

    # Numeric price sources.
    d["_fair_best_bid"] = pd.to_numeric(d["bid_price_1"], errors="coerce")
    d["_fair_best_ask"] = pd.to_numeric(d["ask_price_1"], errors="coerce")

    # Mid source: prefer mid_price if already present, otherwise compute from best bid/ask.
    if "mid_price" in d.columns:
        d["_fair_mid"] = pd.to_numeric(d["mid_price"], errors="coerce")
    elif "mid" in d.columns:
        d["_fair_mid"] = pd.to_numeric(d["mid"], errors="coerce")
    else:
        d["_fair_mid"] = (d["_fair_best_bid"] + d["_fair_best_ask"]) / 2

    # Prefer the actual Prosperity day if present; otherwise use the loaded file index.
    if "day" in d.columns:
        d["_day_label"] = d["day"]
    elif "_file_idx" in d.columns:
        d["_day_label"] = d["_file_idx"]
    else:
        d["_day_label"] = (
            pd.to_numeric(d["timestamp"], errors="coerce") // TIMESTAMP_STRIDE
        ).astype(int)

    # Use local timestamp inside each loaded day, not the globally-offset timestamp.
    d["_local_ts"] = pd.to_numeric(d["timestamp"], errors="coerce") % TIMESTAMP_STRIDE

    d = d.dropna(subset=[
        "_day_label", "_local_ts", "_fair_mid", "_fair_best_bid", "_fair_best_ask"
    ]).copy()

    if d.empty:
        return pd.DataFrame(columns=columns)

    def _aggregate_period(frame: pd.DataFrame, period_name: str) -> pd.DataFrame:
        out = (
            frame
            .groupby("_day_label", sort=False)[
                ["_fair_mid", "_fair_best_bid", "_fair_best_ask"]
            ]
            .mean()
            .reset_index()
            .rename(columns={
                "_day_label": "day",
                "_fair_mid": "fair_mid",
                "_fair_best_bid": "fair_best_bid",
                "_fair_best_ask": "fair_best_ask",
            })
        )
        out["period"] = period_name
        return out

    full_day = _aggregate_period(d, "full_day")
    first_half = _aggregate_period(
        d[d["_local_ts"] < TIMESTAMP_STRIDE / 2],
        "first_half",
    )
    second_half = _aggregate_period(
        d[d["_local_ts"] >= TIMESTAMP_STRIDE / 2],
        "second_half",
    )

    summary = pd.concat([full_day, first_half, second_half], ignore_index=True)
    summary = summary[columns]

    # Sort days sensibly even if the day labels are strings, and force period order.
    period_order = {"full_day": 0, "first_half": 1, "second_half": 2}
    summary["_sort_day"] = pd.to_numeric(summary["day"], errors="coerce")
    summary["_sort_period"] = summary["period"].map(period_order)

    summary = (
        summary
        .sort_values(["_sort_day", "day", "_sort_period"], na_position="last")
        .drop(columns=["_sort_day", "_sort_period"])
        .reset_index(drop=True)
    )

    value_cols = ["fair_mid", "fair_best_bid", "fair_best_ask"]
    summary[value_cols] = summary[value_cols].round(digits)

    return summary


def _draw_halfday_fair_table(ax, summary: pd.DataFrame) -> None:
    """Draw the daily/half-day fair-value table underneath a product chart."""
    ax.axis("off")

    if summary.empty:
        ax.text(
            0.5, 0.5,
            "No fair-value table available",
            ha="center", va="center", fontsize=10,
        )
        return

    display = summary.copy()

    # Clean formatting for notebook display.
    def _fmt_day(x):
        try:
            fx = float(x)
            return str(int(fx)) if fx.is_integer() else str(x)
        except Exception:
            return str(x)

    period_labels = {
        "full_day": "full day",
        "first_half": "first half",
        "second_half": "second half",
    }

    display["day"] = display["day"].map(_fmt_day)
    display["period"] = display["period"].map(lambda x: period_labels.get(x, str(x)))

    for col in ["fair_mid", "fair_best_bid", "fair_best_ask"]:
        display[col] = display[col].map(
            lambda x: "" if pd.isna(x) else f"{float(x):.2f}"
        )

    display = display.rename(columns={
        "day": "day",
        "period": "period",
        "fair_mid": "fair\nmid",
        "fair_best_bid": "fair\nbest bid",
        "fair_best_ask": "fair\nbest ask",
    })

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.20)

    ax.set_title(
        "Mean values by day/period: mid, best bid, best ask",
        fontsize=10,
        pad=4,
    )


def overview(df: pd.DataFrame, window: int = 100) -> None:
    """
    Clean overview:
      - prints dataset summary
      - for each product:
          1) shows one clean price plot
          2) then shows a separate fair-value table

    Fair values are means over:
      - full day
      - first half   : local timestamp < 500_000
      - second half  : local timestamp >= 500_000

    Columns reported:
      - fair_mid      = mean mid_price
      - fair_best_bid = mean bid_price_1
      - fair_best_ask = mean ask_price_1
    """

    # Use notebook HTML tables if available, without pandas .style
    try:
        from IPython.display import display, HTML
        use_html_display = True
    except Exception:
        use_html_display = False

    import html as _html

    products = get_products(df)
    days = sorted(df["_file_idx"].unique()) if "_file_idx" in df.columns else []

    print("=" * 80)
    print(f"Products   : {products}")
    print(f"Days loaded: {len(days)} file(s)")
    print(f"Rows       : {len(df):,}")
    print(f"Timestamps : {int(df['timestamp'].min()):,} → {int(df['timestamp'].max()):,}")
    print("=" * 80)

    if len(products) == 0:
        return

    def _display_clean_table(table_df: pd.DataFrame, title: str) -> None:
        """
        Display a clean HTML table without using pandas Styler,
        so it does NOT require jinja2.
        """
        if not use_html_display:
            print(title)
            print(table_df.to_string(index=False))
            print()
            return

        headers = "".join(
            f"<th>{_html.escape(str(col))}</th>"
            for col in table_df.columns
        )

        body_rows = []
        for _, row in table_df.iterrows():
            cells = "".join(
                f"<td>{_html.escape(str(value))}</td>"
                for value in row
            )
            body_rows.append(f"<tr>{cells}</tr>")

        body = "\n".join(body_rows)

        html_table = f"""
        <div style="margin-top: 10px; margin-bottom: 35px;">
            <h3 style="font-family: Arial, sans-serif; margin-bottom: 8px;">
                { _html.escape(title) }
            </h3>

            <table style="
                border-collapse: collapse;
                font-family: Arial, sans-serif;
                font-size: 13px;
                min-width: 650px;
                max-width: 900px;
            ">
                <thead>
                    <tr style="background-color: #f2f2f2;">
                        {headers}
                    </tr>
                </thead>
                <tbody>
                    {body}
                </tbody>
            </table>
        </div>

        <style>
            table th {{
                border: 1px solid #999;
                padding: 6px 10px;
                text-align: center;
                font-weight: bold;
            }}
            table td {{
                border: 1px solid #ccc;
                padding: 6px 10px;
                text-align: center;
            }}
        </style>
        """

        display(HTML(html_table))

    for product in products:
        print("\n" + "=" * 100)
        print(product)
        print("=" * 100)

        # ─────────────────────────────────────────────
        # 1) Clean price plot
        # ─────────────────────────────────────────────
        try:
            p = enrich(df, product, window)

            plt.figure(figsize=(22, 6))
            plt.plot(
                p["timestamp"],
                p["mid_price"],
                linewidth=1.5,
                label="Mid Price",
            )
            plt.plot(
                p["timestamp"],
                p["ma"],
                linewidth=1.2,
                linestyle="--",
                label=f"MA({window})",
            )

            plt.title(f"{product} — Price Overview", fontsize=15)
            plt.xlabel("Timestamp")
            plt.ylabel("Mid Price")
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        except Exception as exc:
            print(f"Could not plot {product}: {exc}")
            continue

        # ─────────────────────────────────────────────
        # 2) Separate fair-value table
        # ─────────────────────────────────────────────
        try:
            fair_df = p.copy()

            fair_df["fair_mid"] = pd.to_numeric(fair_df["mid_price"], errors="coerce")
            fair_df["fair_best_bid"] = pd.to_numeric(fair_df["bid_price_1"], errors="coerce")
            fair_df["fair_best_ask"] = pd.to_numeric(fair_df["ask_price_1"], errors="coerce")

            # Prefer actual day column if available
            if "day" in fair_df.columns:
                fair_df["day_label"] = fair_df["day"]
            elif "_file_idx" in fair_df.columns:
                fair_df["day_label"] = fair_df["_file_idx"]
            else:
                fair_df["day_label"] = fair_df["timestamp"] // TIMESTAMP_STRIDE

            # Local timestamp inside each day
            fair_df["local_timestamp"] = fair_df["timestamp"] % TIMESTAMP_STRIDE

            fair_df["half_period"] = np.where(
                fair_df["local_timestamp"] < 500_000,
                "first half",
                "second half",
            )

            rows = []

            for day, g_day in fair_df.groupby("day_label", sort=True):
                rows.append({
                    "day": day,
                    "period": "full day",
                    "fair_mid": g_day["fair_mid"].mean(),
                    "fair_best_bid": g_day["fair_best_bid"].mean(),
                    "fair_best_ask": g_day["fair_best_ask"].mean(),
                })

                for period in ["first half", "second half"]:
                    g = g_day[g_day["half_period"] == period]

                    rows.append({
                        "day": day,
                        "period": period,
                        "fair_mid": g["fair_mid"].mean(),
                        "fair_best_bid": g["fair_best_bid"].mean(),
                        "fair_best_ask": g["fair_best_ask"].mean(),
                    })

            fair_table = pd.DataFrame(rows)

            period_order = pd.CategoricalDtype(
                ["full day", "first half", "second half"],
                ordered=True,
            )

            fair_table["period"] = fair_table["period"].astype(period_order)

            fair_table = (
                fair_table
                .sort_values(["day", "period"])
                .reset_index(drop=True)
            )

            fair_table[["fair_mid", "fair_best_bid", "fair_best_ask"]] = (
                fair_table[["fair_mid", "fair_best_bid", "fair_best_ask"]]
                .round(2)
            )

            _display_clean_table(
                fair_table,
                title=f"{product} — Fair values by day / period"
            )

        except Exception as exc:
            print(f"Could not build fair-value table for {product}: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# 7 · ORDER BOOK ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def plot_orderbook_levels(
    df: pd.DataFrame,
    product: str,
    ts_start: Optional[int] = None,
    ts_end: Optional[int] = None,
    interactive: bool = True,
) -> None:
    """
    Plot all three bid and ask price levels over time.

    Bids in blue shades, asks in red shades. Thicker/darker = level 1 (best).
    This replicates the core view from the Frankfurt Hedgehogs dashboard and lets
    you visually identify:
      - Wall prices: stable, deep levels that barely move (the market maker anchor)
      - Noise quotes: thin levels that jump around
      - Moments when the spread widens or tightens

    Parameters
    ----------
    ts_start / ts_end : optional timestamp slice for zooming in
    """
    p = df[df["product"] == product].copy()
    if ts_start: p = p[p["timestamp"] >= ts_start]
    if ts_end:   p = p[p["timestamp"] <= ts_end]

    bid_levels = [("bid_price_1", "#1f77b4", 2.0),
                  ("bid_price_2", "#6baed6", 1.2),
                  ("bid_price_3", "#bdd7e7", 0.7)]
    ask_levels = [("ask_price_1", "#d62728", 2.0),
                  ("ask_price_2", "#fc8d59", 1.2),
                  ("ask_price_3", "#fee090", 0.7)]

    t = f"{product} — Order Book Levels"

    if interactive and PLOTLY:
        fig = go.Figure()
        for col, color, width in bid_levels + ask_levels:
            if col in p.columns:
                fig.add_trace(go.Scatter(
                    x=p["timestamp"], y=pd.to_numeric(p[col], errors="coerce"),
                    name=col, line=dict(color=color, width=width), opacity=0.8,
                ))
        fig.update_layout(title=t, template="plotly_white", height=500,
                          xaxis_title="Timestamp", yaxis_title="Price",
                          legend=dict(orientation="h", y=1.05))
        fig.show()
    else:
        plt.figure(figsize=(22, 7))
        for col, color, lw in bid_levels + ask_levels:
            if col in p.columns:
                vals = pd.to_numeric(p[col], errors="coerce")
                plt.plot(p["timestamp"], vals, color=color, linewidth=lw,
                         label=col, alpha=0.8)
        plt.title(t)
        plt.xlabel("Timestamp")
        plt.ylabel("Price")
        plt.legend(ncol=3, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def compute_book_imbalance(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """
    Compute order book imbalance at every timestamp.

    imbalance = (total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol)

    Ranges from -1 (all liquidity on ask side) to +1 (all on bid side).
    A consistently positive imbalance → buy pressure → price likely to rise.
    Negative → sell pressure.

    This is a short-term directional signal used heavily by HFTs.
    For mean-reverting products it can help time entries/exits.

    Returns the product DataFrame with an added 'imbalance' column.
    """
    p = df[df["product"] == product].copy()
    bid_vol_cols = ["bid_volume_1", "bid_volume_2", "bid_volume_3"]
    ask_vol_cols = ["ask_volume_1", "ask_volume_2", "ask_volume_3"]

    for c in bid_vol_cols + ask_vol_cols:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)

    p["total_bid_vol"] = p[bid_vol_cols].sum(axis=1)
    p["total_ask_vol"] = p[ask_vol_cols].sum(axis=1)
    total = p["total_bid_vol"] + p["total_ask_vol"]
    p["imbalance"] = (p["total_bid_vol"] - p["total_ask_vol"]) / total.replace(0, np.nan)
    return p


def plot_book_imbalance(
    df: pd.DataFrame,
    product: str,
    window: int = 50,
    interactive: bool = True,
) -> None:
    """
    Plot book imbalance over time with mid-price on a secondary axis.

    Look for:
      - Sustained imbalance in one direction → precedes price move
      - Imbalance reversals → potential mean-reversion entry signal
    """
    p = compute_book_imbalance(df, product)
    p["mid_price"]      = (pd.to_numeric(p["ask_price_1"], errors="coerce") +
                           pd.to_numeric(p["bid_price_1"], errors="coerce")) / 2
    p["imbalance_ma"]   = p["imbalance"].rolling(window).mean()

    t = f"{product} — Book Imbalance (window={window})"

    if interactive and PLOTLY:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["imbalance"],
                                 name="Imbalance", line=dict(color="steelblue", width=1),
                                 opacity=0.4), secondary_y=False)
        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["imbalance_ma"],
                                 name=f"Imbalance MA({window})",
                                 line=dict(color="navy", width=2)), secondary_y=False)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        fig.add_trace(go.Scatter(x=p["timestamp"], y=p["mid_price"],
                                 name="Mid Price", line=dict(color="darkorange", width=1.5),
                                 opacity=0.6), secondary_y=True)
        fig.update_layout(title=t, template="plotly_white", height=500,
                          xaxis=dict(domain=[0, 0.94]))
        fig.update_yaxes(title_text="Imbalance [-1, +1]", secondary_y=False)
        fig.update_yaxes(title_text="Mid Price",           secondary_y=True)
        fig.show()
    else:
        fig, ax1 = plt.subplots(figsize=(22, 7))
        ax2 = ax1.twinx()
        ax1.plot(p["timestamp"], p["imbalance"],    color="steelblue", linewidth=1,
                 alpha=0.4, label="Imbalance")
        ax1.plot(p["timestamp"], p["imbalance_ma"], color="navy",      linewidth=2,
                 label=f"MA({window})")
        ax1.axhline(0, linestyle=":", color="gray", alpha=0.5)
        ax1.set_ylabel("Imbalance [-1, +1]", color="navy")
        ax2.plot(p["timestamp"], p["mid_price"], color="darkorange",
                 linewidth=1.5, alpha=0.6, label="Mid Price")
        ax2.set_ylabel("Mid Price", color="darkorange")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines])
        plt.title(t)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_return_autocorrelation(
    df: pd.DataFrame,
    product: str,
    max_lag: int = 20,
    n_random: int = 100,
    interactive: bool = True,
) -> None:
    """
    Plot autocorrelation of mid-price returns at lags 1 … max_lag.

    Overlays a random baseline: autocorrelations computed from n_random shuffled
    (randomised) versions of the same return series. The grey band shows the
    range you'd expect from pure noise.

    What to look for:
      - Lag-1 strongly NEGATIVE → mean reversion (Squid Ink, Volcanic Rock in P3)
      - Lag-1 strongly POSITIVE → momentum / trending
      - All lags inside the grey band → random walk (no exploitable structure)

    This was the key statistical test the Frankfurt Hedgehogs used to confirm
    mean-reversion before committing to that strategy.
    """
    p = enrich(df, product, window=1)
    returns = p["mid_price"].diff().dropna().values

    # Compute autocorrelation at each lag
    def acf(series, lag):
        s = series - series.mean()
        return np.corrcoef(s[:-lag], s[lag:])[0, 1] if lag < len(s) else np.nan

    lags      = list(range(1, max_lag + 1))
    acf_vals  = [acf(returns, lag) for lag in lags]

    # Random baseline: shuffle returns n_random times
    random_acfs = []
    rng = np.random.default_rng(42)
    for _ in range(n_random):
        shuffled = rng.permutation(returns)
        random_acfs.append([acf(shuffled, lag) for lag in lags])
    rand_arr = np.array(random_acfs)
    rand_lo  = np.percentile(rand_arr, 5,  axis=0)
    rand_hi  = np.percentile(rand_arr, 95, axis=0)

    t = f"{product} — Return Autocorrelation (lags 1–{max_lag})"

    if interactive and PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lags + lags[::-1],
            y=list(rand_hi) + list(rand_lo[::-1]),
            fill="toself", fillcolor="rgba(128,128,128,0.2)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Random 90% band",
        ))
        fig.add_trace(go.Bar(x=lags, y=acf_vals, name="ACF",
                             marker_color=["#d62728" if v < 0 else "#1f77b4"
                                           for v in acf_vals]))
        fig.add_hline(y=0, line_dash="dot", line_color="black", opacity=0.5)
        fig.update_layout(title=t, template="plotly_white", height=450,
                          xaxis_title="Lag", yaxis_title="Autocorrelation")
        fig.show()
    else:
        plt.figure(figsize=(14, 5))
        plt.fill_between(lags, rand_lo, rand_hi, alpha=0.2,
                         color="gray", label="Random 90% band")
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in acf_vals]
        plt.bar(lags, acf_vals, color=colors, label="ACF")
        plt.axhline(0, linestyle=":", color="black", alpha=0.5)
        plt.title(t)
        plt.xlabel("Lag")
        plt.ylabel("Autocorrelation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def trade_size_distribution(
    df_trades: pd.DataFrame,
    symbol: str,
    interactive: bool = True,
) -> pd.DataFrame:
    """
    Show the distribution of trade sizes for a given product.

    Key use: detecting informed/systematic traders by their signature size.
    The Frankfurt Hedgehogs found Olivia purely by noticing a consistent 15-lot
    size at daily extrema — visible as a spike in this histogram.

    Returns a summary DataFrame of size → count.
    """
    sub = df_trades[df_trades["symbol"] == symbol].copy()
    sub["quantity"] = pd.to_numeric(sub["quantity"], errors="coerce").abs()
    counts = (sub["quantity"].value_counts()
                             .sort_index()
                             .reset_index()
                             .rename(columns={"index": "size", "quantity": "count"}))
    # newer pandas returns columns named differently
    counts.columns = ["size", "count"]

    t = f"{symbol} — Trade Size Distribution"

    if interactive and PLOTLY:
        fig = go.Figure(go.Bar(x=counts["size"], y=counts["count"],
                               marker_color="#1f77b4"))
        fig.update_layout(title=t, template="plotly_white", height=400,
                          xaxis_title="Trade Size", yaxis_title="Count")
        fig.show()
    else:
        plt.figure(figsize=(14, 4))
        plt.bar(counts["size"], counts["count"], color="#1f77b4")
        plt.title(t)
        plt.xlabel("Trade Size")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    print(f"\nTop 10 most common trade sizes for {symbol}:")
    print(counts.nlargest(10, "count").to_string(index=False))
    return counts


def find_extrema_trades(
    df_prices: pd.DataFrame,
    df_trades: pd.DataFrame,
    symbol: str,
    qty_filter: Optional[int] = None,
    tolerance: float = 0.002,
) -> pd.DataFrame:
    """
    Find trades that occurred near the daily high or daily low price.

    This is the core analysis that would have revealed "Olivia" in Prosperity 3:
    she consistently bought at the daily minimum and sold at the daily maximum,
    always in lots of 15. A function like this, run on historical data, would
    have exposed this pattern before Round 1 even ended.

    Parameters
    ----------
    qty_filter  : if set, only consider trades of exactly this size
    tolerance   : how close to the daily extreme counts as "at the extreme"
                  (fraction of daily range, default 0.2%)

    Returns
    -------
    DataFrame of flagged trades with columns: timestamp, price, quantity,
    buyer, seller, day_low, day_high, flag ('low' | 'high' | None)
    """
    # Build daily price range from prices file
    p = df_prices[df_prices["product"] == symbol].copy()
    p["mid"] = (pd.to_numeric(p["ask_price_1"], errors="coerce") +
                pd.to_numeric(p["bid_price_1"], errors="coerce")) / 2
    p["day"] = p["timestamp"] // TIMESTAMP_STRIDE

    daily = p.groupby("day")["mid"].agg(day_low="min", day_high="max").reset_index()

    t = df_trades[df_trades["symbol"] == symbol].copy()
    t["quantity"] = pd.to_numeric(t["quantity"], errors="coerce").abs()
    t["price"]    = pd.to_numeric(t["price"],    errors="coerce")
    t["day"]      = t["timestamp"] // TIMESTAMP_STRIDE

    if qty_filter is not None:
        t = t[t["quantity"] == qty_filter]

    t = t.merge(daily, on="day", how="left")

    daily_range = (t["day_high"] - t["day_low"]).replace(0, np.nan)
    tol_abs     = tolerance * daily_range

    t["flag"] = None
    t.loc[t["price"] <= t["day_low"] + tol_abs, "flag"] = "low"
    t.loc[t["price"] >= t["day_high"] - tol_abs, "flag"] = "high"

    flagged = t[t["flag"].notna()].copy()
    print(f"Trades near daily extrema for {symbol}"
          + (f" (qty={qty_filter})" if qty_filter else "") + ":")
    print(f"  Near daily LOW  : {(flagged['flag']=='low').sum()}")
    print(f"  Near daily HIGH : {(flagged['flag']=='high').sum()}")
    if "buyer" in flagged.columns:
        print(f"\nBuyers  at daily LOW  :", flagged[flagged["flag"]=="low"]["buyer"].value_counts().head(5).to_dict())
        print(f"Sellers at daily HIGH :", flagged[flagged["flag"]=="high"]["seller"].value_counts().head(5).to_dict())
    return flagged


def compute_fill_edge(
    df_prices: pd.DataFrame,
    df_trades: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    For each trade, compute the 'edge': how far the trade price deviated from
    the concurrent mid-price.

    edge = trade_price - mid_price

    Positive edge on a buy = buyer paid above mid (taker, possibly uninformed).
    Negative edge on a buy = buyer got below mid (maker, or very informed).
    Large absolute edge = someone crossed a lot of spread → potential informed trade.

    This is useful for:
      - Identifying which trader types pay the most/least edge
      - Calibrating your own taker logic: how much edge are you giving away?
      - Spotting aggressive institutional-style flow
    """
    # Align mid-price to trade timestamps using nearest merge
    p = df_prices[df_prices["product"] == symbol][["timestamp", "ask_price_1", "bid_price_1"]].copy()
    p["mid"] = (pd.to_numeric(p["ask_price_1"], errors="coerce") +
                pd.to_numeric(p["bid_price_1"], errors="coerce")) / 2
    p = p[["timestamp", "mid"]].dropna()

    t = df_trades[df_trades["symbol"] == symbol].copy()
    t["price"]    = pd.to_numeric(t["price"],    errors="coerce")
    t["quantity"] = pd.to_numeric(t["quantity"], errors="coerce").abs()

    # Merge on nearest timestamp
    merged = pd.merge_asof(
        t.sort_values("timestamp"),
        p.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
    )
    merged["edge"]     = merged["price"] - merged["mid"]
    merged["abs_edge"] = merged["edge"].abs()

    print(f"Fill edge summary for {symbol}:")
    print(f"  Mean |edge| : {merged['abs_edge'].mean():.4f}")
    print(f"  Max  |edge| : {merged['abs_edge'].max():.4f}")
    print(f"  Median edge : {merged['edge'].median():.4f}")
    return merged


def plot_fill_edge(
    df_prices: pd.DataFrame,
    df_trades: pd.DataFrame,
    symbol: str,
    interactive: bool = True,
) -> None:
    """
    Plot the distribution of fill edges and edge over time.
    See compute_fill_edge for definition.
    """
    m = compute_fill_edge(df_prices, df_trades, symbol)

    if interactive and PLOTLY:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Edge Distribution", "Edge Over Time"))
        fig.add_trace(go.Histogram(x=m["edge"], nbinsx=60,
                                   marker_color="#1f77b4", name="Edge"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=m["timestamp"], y=m["edge"],
                                 mode="markers", marker=dict(size=3, color="#d62728",
                                 opacity=0.5), name="Edge over time"),
                      row=1, col=2)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)
        fig.update_layout(title=f"{symbol} — Fill Edge Analysis",
                          template="plotly_white", height=450, showlegend=False)
        fig.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
        ax1.hist(m["edge"].dropna(), bins=60, color="#1f77b4", edgecolor="white")
        ax1.set_title("Edge Distribution")
        ax1.set_xlabel("Edge")
        ax2.scatter(m["timestamp"], m["edge"], s=5, color="#d62728", alpha=0.4)
        ax2.axhline(0, linestyle=":", color="gray")
        ax2.set_title("Edge Over Time")
        ax2.set_xlabel("Timestamp")
        fig.suptitle(f"{symbol} — Fill Edge Analysis")
        plt.tight_layout()
        plt.show()