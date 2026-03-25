import json
import os

import numpy as np
import optuna
from autogluon.tabular import TabularPredictor
from catboost import CatBoostRegressor, Pool

from src.config import (
    N_CPUS,
    USE_GPU,
    STEP1_DIR,
    STEP2_DIR,
    AUTOGLOUON_HYPERPARAMS,
)
from src.utils import conservative_postprocess, dist_mean


CAT_COLS = [
    "team_id_last",
    "player_id_last",
    "action_id_last",
    "result_name_last",
    "x_zone",
    "y_zone",
    "goal_dist_zone",
    "has_enough_hist",
    "type_name_last",
]


def fit_autogluon(train_x, train_y, val_x, val_y, label, path):
    os.makedirs(path, exist_ok=True)
    tr = train_x.copy()
    tr[label] = train_y
    va = val_x.copy()
    va[label] = val_y
    return TabularPredictor(
        label=label,
        problem_type="regression",
        eval_metric="mae",
        path=path,
    ).fit(
        train_data=tr,
        tuning_data=va,
        presets="medium_quality",
        hyperparameters=AUTOGLOUON_HYPERPARAMS,
        num_cpus=N_CPUS,
        verbosity=2,
    )


def load_step1_predictor(fold_idx: int, axis: str):
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    label_name = "delta_x" if axis == "x" else "delta_y"
    model_path = os.path.join(STEP1_DIR, f"fold{fold_idx + 1}_{label_name}")
    return TabularPredictor.load(model_path)


def force_cat_dtypes(df):
    df = df.copy()
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def align_columns(df, col_list):
    df = df.copy()
    for c in col_list:
        if c not in df.columns:
            df[c] = np.nan
    return df[col_list]


def catboost_pool(x, y=None):
    x = force_cat_dtypes(x)
    cat_cols_in_x = [c for c in CAT_COLS if c in x.columns]
    return Pool(x, label=y, cat_features=cat_cols_in_x)


def suggest_cat_params(trial: optuna.Trial, seed: int):
    params = {
        "random_seed": seed,
        "thread_count": N_CPUS,
        "verbose": False,
        "allow_writing_files": False,
        "loss_function": "MAE",
        "iterations": trial.suggest_int("iterations", 400, 1600),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 30),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide"]),
        "od_type": "Iter",
        "od_wait": trial.suggest_int("od_wait", 20, 60),
    }
    if USE_GPU:
        params["task_type"] = "GPU"
    return params


def tune_step2_separate_studies_for_fold(
    fold_idx: int,
    x_train_res_x,
    res_train_dx,
    x_train_res_y,
    res_train_dy,
    x_valid_res_x,
    base_valid_dx,
    x_valid_res_y,
    base_valid_dy,
    valid_start_x,
    valid_start_y,
    y_valid_abs_x,
    y_valid_abs_y,
    context_valid_df,
    seed: int,
    trials_dx: int,
    trials_dy: int,
):
    feat_cols_x = list(x_train_res_x.columns)
    feat_cols_y = list(x_train_res_y.columns)

    x_train_res_x = align_columns(x_train_res_x, feat_cols_x)
    x_valid_res_x = align_columns(x_valid_res_x, feat_cols_x)
    x_train_res_y = align_columns(x_train_res_y, feat_cols_y)
    x_valid_res_y = align_columns(x_valid_res_y, feat_cols_y)

    best_dx = {
        "best_dist": 1e18,
        "alpha_x": 0.0,
        "params": None,
        "model": None,
    }

    def obj_dx(trial: optuna.Trial):
        alpha_x = trial.suggest_float("alpha_x", 0.0, 1.0)
        params = suggest_cat_params(trial, seed + 11)

        m_dx = CatBoostRegressor(**params)
        pool_tr_x = catboost_pool(x_train_res_x, res_train_dx)
        pool_va_x = catboost_pool(x_valid_res_x, None)
        m_dx.fit(pool_tr_x, eval_set=pool_va_x)

        rdx = m_dx.predict(pool_va_x).astype(np.float32)

        pred_x = valid_start_x + (base_valid_dx + alpha_x * rdx)
        pred_y = valid_start_y + (base_valid_dy + 0.0)

        pred_x, pred_y = conservative_postprocess(pred_x, pred_y, context_valid_df)
        d = dist_mean(pred_x, pred_y, y_valid_abs_x, y_valid_abs_y)

        if d < best_dx["best_dist"]:
            best_dx["best_dist"] = float(d)
            best_dx["alpha_x"] = float(alpha_x)
            best_dx["params"] = params
            best_dx["model"] = m_dx

        return float(d)

    print(f"\n[Step2-DX] Optuna tuning (fold {fold_idx + 1}) | trials={trials_dx}")
    sampler_dx = optuna.samplers.TPESampler(seed=seed + 111)
    study_dx = optuna.create_study(direction="minimize", sampler=sampler_dx)
    study_dx.optimize(obj_dx, n_trials=trials_dx, show_progress_bar=False)
    print(f"[Step2-DX] best_dist={best_dx['best_dist']:.4f} alpha_x={best_dx['alpha_x']:.3f}")

    rdx_best = best_dx["model"].predict(catboost_pool(x_valid_res_x)).astype(np.float32)
    alpha_x_best = best_dx["alpha_x"]

    best_dy = {
        "best_dist": 1e18,
        "alpha_y": 0.0,
        "params": None,
        "model": None,
    }

    def obj_dy(trial: optuna.Trial):
        alpha_y = trial.suggest_float("alpha_y", 0.0, 1.0)
        params = suggest_cat_params(trial, seed + 22)

        m_dy = CatBoostRegressor(**params)
        pool_tr_y = catboost_pool(x_train_res_y, res_train_dy)
        pool_va_y = catboost_pool(x_valid_res_y, None)
        m_dy.fit(pool_tr_y, eval_set=pool_va_y)

        rdy = m_dy.predict(pool_va_y).astype(np.float32)

        pred_x = valid_start_x + (base_valid_dx + alpha_x_best * rdx_best)
        pred_y = valid_start_y + (base_valid_dy + alpha_y * rdy)

        pred_x, pred_y = conservative_postprocess(pred_x, pred_y, context_valid_df)
        d = dist_mean(pred_x, pred_y, y_valid_abs_x, y_valid_abs_y)

        if d < best_dy["best_dist"]:
            best_dy["best_dist"] = float(d)
            best_dy["alpha_y"] = float(alpha_y)
            best_dy["params"] = params
            best_dy["model"] = m_dy

        return float(d)

    print(f"\n[Step2-DY] Optuna tuning (fold {fold_idx + 1}) | trials={trials_dy}")
    sampler_dy = optuna.samplers.TPESampler(seed=seed + 222)
    study_dy = optuna.create_study(direction="minimize", sampler=sampler_dy)
    study_dy.optimize(obj_dy, n_trials=trials_dy, show_progress_bar=False)
    print(f"[Step2-DY] best_dist={best_dy['best_dist']:.4f} alpha_y={best_dy['alpha_y']:.3f}")

    rdy_best = best_dy["model"].predict(catboost_pool(x_valid_res_y)).astype(np.float32)
    alpha_y_best = best_dy["alpha_y"]

    pred_x = valid_start_x + (base_valid_dx + alpha_x_best * rdx_best)
    pred_y = valid_start_y + (base_valid_dy + alpha_y_best * rdy_best)
    pred_x, pred_y = conservative_postprocess(pred_x, pred_y, context_valid_df)
    final_dist = dist_mean(pred_x, pred_y, y_valid_abs_x, y_valid_abs_y)

    return {
        "best_dist": float(final_dist),
        "alpha_x": float(alpha_x_best),
        "alpha_y": float(alpha_y_best),
        "best_params_dx": best_dx["params"],
        "best_params_dy": best_dy["params"],
        "best_model_dx": best_dx["model"],
        "best_model_dy": best_dy["model"],
        "feat_cols_x": feat_cols_x,
        "feat_cols_y": feat_cols_y,
    }


def save_step2_pack(fold_idx: int, pack: dict):
    fold_dir = os.path.join(STEP2_DIR, f"fold{fold_idx + 1}")
    os.makedirs(fold_dir, exist_ok=True)

    dx_path = os.path.join(fold_dir, "cat_residual_dx.cbm")
    dy_path = os.path.join(fold_dir, "cat_residual_dy.cbm")
    pack["best_model_dx"].save_model(dx_path)
    pack["best_model_dy"].save_model(dy_path)

    meta = {
        "best_dist": pack["best_dist"],
        "alpha_x": pack["alpha_x"],
        "alpha_y": pack["alpha_y"],
        "best_params_dx": pack["best_params_dx"],
        "best_params_dy": pack["best_params_dy"],
        "feat_cols_x": pack["feat_cols_x"],
        "feat_cols_y": pack["feat_cols_y"],
        "cat_cols": [c for c in CAT_COLS],
        "note": (
            "DX/DY separate Optuna studies; alpha_x/alpha_y are tuned as trial params "
            "(no grid calibration). Best trial models are saved directly (no refit)."
        ),
    }
    with open(os.path.join(fold_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_step2_pack(fold_idx: int):
    fold_dir = os.path.join(STEP2_DIR, f"fold{fold_idx + 1}")
    meta_path = os.path.join(fold_dir, "meta.json")
    dx_path = os.path.join(fold_dir, "cat_residual_dx.cbm")
    dy_path = os.path.join(fold_dir, "cat_residual_dy.cbm")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    m_dx = CatBoostRegressor()
    m_dy = CatBoostRegressor()
    m_dx.load_model(dx_path)
    m_dy.load_model(dy_path)

    return {
        "m_dx": m_dx,
        "m_dy": m_dy,
        "alpha_x": float(meta["alpha_x"]),
        "alpha_y": float(meta["alpha_y"]),
        "feat_cols_x": meta["feat_cols_x"],
        "feat_cols_y": meta["feat_cols_y"],
    }
