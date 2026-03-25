import os
import random
import warnings

import numpy as np
import pandas as pd

from src.config import (
    N_CPUS,
    RANDOM_SEED,
    STEP1_DIR,
    STEP2_DIR,
    BASE_OUTPUT_DIR,
    POSTPROC_LONGPASS_BOOST,
    LONGPASS_BOOST_GOAL_DIST_ZONE,
    LONGPASS_BOOST_PRED_DIST_THRESH,
    LONGPASS_BOOST_SCALE,
)


def configure_environment():
    warnings.filterwarnings("ignore")
    for k in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[k] = str(N_CPUS)

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(STEP1_DIR, exist_ok=True)
    os.makedirs(STEP2_DIR, exist_ok=True)


def replace_inf(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns and df[c].dtype.kind in "fc":
            df[c].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def merge_stats_fillna(df: pd.DataFrame, stats_df: pd.DataFrame, left_key: str, right_key: str, drop_key: str):
    if stats_df is None:
        return df
    out = df.merge(stats_df, left_on=left_key, right_on=right_key, how="left", suffixes=("", "_dup"))
    if drop_key in out.columns:
        out = out.drop(drop_key, axis=1)
    stat_cols = [c for c in stats_df.columns if c != right_key]
    for c in stat_cols:
        if c in out.columns:
            out[c] = out[c].fillna(out[c].mean())
    return out


def log1p_safe(x):
    x = x.astype(float)
    return np.sign(x) * np.log1p(np.abs(x))


def add_norm_coords(df: pd.DataFrame):
    coord_cols = [
        "start_x_last", "start_y_last", "start_x_first", "start_y_first",
        "prev_end_x", "prev_end_y", "recent3_mean_x", "recent3_mean_y",
    ]
    for c in coord_cols:
        if c not in df.columns:
            continue
        if "x" in c:
            df[c + "_norm"] = df[c] / 105.0
        elif "y" in c:
            df[c + "_norm"] = df[c] / 68.0
    return df


def apply_log_transform(df: pd.DataFrame):
    log_cols = [
        c for c in df.columns
        if (
            c.startswith("cnt_")
            or c.endswith("_len")
            or c.endswith("_time")
            or c in ["n_events", "n_hist_events", "pass_chain_length"]
        )
    ]
    for c in log_cols:
        if c in df.columns:
            df[c + "_log1p"] = log1p_safe(df[c])
    return df


def clip_outliers(df: pd.DataFrame, cols=None, low_q=0.01, high_q=0.99):
    cols = [c for c in df.columns if df[c].dtype.kind in "fc"] if cols is None else cols
    for c in cols:
        q_low, q_high = df[c].quantile(low_q), df[c].quantile(high_q)
        df[c] = df[c].clip(q_low, q_high)
    return df


def cast_categoricals(df: pd.DataFrame):
    cat_cols = [
        "team_id_last", "player_id_last", "action_id_last", "result_name_last",
        "x_zone", "y_zone", "goal_dist_zone", "has_enough_hist", "type_name_last",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def conservative_postprocess(pred_x, pred_y, context_df):
    pred_x = np.asarray(pred_x, dtype=float)
    pred_y = np.asarray(pred_y, dtype=float)

    if POSTPROC_LONGPASS_BOOST and (context_df is not None):
        need = {"goal_dist_zone", "start_x_last", "start_y_last"}
        if need.issubset(context_df.columns):
            start_x = context_df["start_x_last"].astype(float).values
            start_y = context_df["start_y_last"].astype(float).values
            gzone = pd.to_numeric(context_df["goal_dist_zone"], errors="coerce").fillna(-1).astype(int).values

            dx, dy = pred_x - start_x, pred_y - start_y
            pred_dist = np.sqrt(dx**2 + dy**2)

            cond = (gzone == LONGPASS_BOOST_GOAL_DIST_ZONE) & (pred_dist >= LONGPASS_BOOST_PRED_DIST_THRESH)
            scale = np.where(cond, LONGPASS_BOOST_SCALE, 1.0)
            pred_x = start_x + dx * scale
            pred_y = start_y + dy * scale

    return np.clip(pred_x, 0.0, 105.0), np.clip(pred_y, 0.0, 68.0)


def dist_mean(px, py, tx, ty):
    return float(np.sqrt((px - tx) ** 2 + (py - ty) ** 2).mean())
