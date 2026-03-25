import json

import pandas as pd

from src.config import (
    TRAIN_PATH,
    MATCH_INFO_PATH,
    RANDOM_SEED,
    RUN_FOLDS,
    N_FOLDS,
    OPTUNA_TRIALS_DX,
    OPTUNA_TRIALS_DY,
    USE_PLAYER_FEATURES,
    USE_TEAM_FEATURES,
    PIPELINE_META_PATH,
    FOLD_RESULTS_PATH,
    STEP1_DIR,
)
from src.features import build_train_base, fold_game_ids, setup_game_date_cv, split_xy
from src.model_utils import (
    fit_autogluon,
    tune_step2_separate_studies_for_fold,
    save_step2_pack,
    align_columns,
    catboost_pool,
)
from src.stats import compute_player_statistics_simple, compute_team_statistics_simple
from src.utils import (
    configure_environment,
    merge_stats_fillna,
    replace_inf,
    conservative_postprocess,
    dist_mean,
)


def train_pipeline():
    configure_environment()

    print("=" * 80)
    print("Train: Step1 AutoGluon (delta) + Step2 CatBoost residual")
    print("=" * 80)

    print("\nLoading data...")
    train_df_raw = pd.read_csv(TRAIN_PATH)
    match_info = pd.read_csv(MATCH_INFO_PATH)
    match_info["game_date"] = pd.to_datetime(match_info["game_date"])

    sorted_games, val_games_per_fold, start_val_region = setup_game_date_cv(train_df_raw, match_info)

    print("\nBuilding base features (train)...")
    x_all_base = build_train_base(train_df_raw)
    episode_meta = x_all_base[["game_episode", "game_id"]].merge(
        match_info[["game_id", "game_date"]],
        on="game_id",
        how="left",
    )

    fold_results = []
    global_feature_cols_x = None
    global_feature_cols_y = None

    for fold_idx in range(N_FOLDS):
        if fold_idx not in RUN_FOLDS:
            print("\n" + "=" * 70)
            print(f"Skipping Fold {fold_idx + 1}/{N_FOLDS}")
            continue

        val_game_ids = fold_game_ids(sorted_games, val_games_per_fold, start_val_region, fold_idx)
        episode_meta["is_valid_game"] = episode_meta["game_id"].isin(val_game_ids)
        valid_mask = episode_meta["is_valid_game"].values
        train_mask = ~valid_mask

        train_game_ids = set(episode_meta.loc[train_mask, "game_id"].unique())
        train_df_fold = train_df_raw[train_df_raw["game_id"].isin(train_game_ids)]

        player_stats_fold = compute_player_statistics_simple(train_df_fold, 10) if USE_PLAYER_FEATURES else None
        team_stats_fold = compute_team_statistics_simple(train_df_fold, 50) if USE_TEAM_FEATURES else None

        x_all = x_all_base.copy()
        if USE_PLAYER_FEATURES:
            x_all = merge_stats_fillna(x_all, player_stats_fold, "player_id_last", "player_id", "player_id")
        if USE_TEAM_FEATURES:
            x_all = merge_stats_fillna(x_all, team_stats_fold, "team_id_last", "team_id", "team_id")

        x_train_fold = x_all.loc[train_mask].reset_index(drop=True)
        x_valid_fold = x_all.loc[valid_mask].reset_index(drop=True)

        y_train_dx = x_train_fold["label_delta_x"].values.astype(float)
        y_train_dy = x_train_fold["label_delta_y"].values.astype(float)
        y_valid_dx = x_valid_fold["label_delta_x"].values.astype(float)
        y_valid_dy = x_valid_fold["label_delta_y"].values.astype(float)

        y_valid_abs_x = x_valid_fold["label_end_x_abs"].values.astype(float)
        y_valid_abs_y = x_valid_fold["label_end_y_abs"].values.astype(float)
        valid_start_x = x_valid_fold["start_x_last"].values.astype(float)
        valid_start_y = x_valid_fold["start_y_last"].values.astype(float)

        feature_cols_x, feature_cols_y = split_xy(x_all)
        if global_feature_cols_x is None:
            global_feature_cols_x = feature_cols_x
            global_feature_cols_y = feature_cols_y

        x_train_feat_x = x_train_fold[[c for c in feature_cols_x if c in x_train_fold.columns]].copy()
        x_valid_feat_x = x_valid_fold[[c for c in feature_cols_x if c in x_valid_fold.columns]].copy()
        x_train_feat_y = x_train_fold[[c for c in feature_cols_y if c in x_train_fold.columns]].copy()
        x_valid_feat_y = x_valid_fold[[c for c in feature_cols_y if c in x_valid_fold.columns]].copy()

        replace_inf(x_train_feat_x, feature_cols_x)
        replace_inf(x_valid_feat_x, feature_cols_x)
        replace_inf(x_train_feat_y, feature_cols_y)
        replace_inf(x_valid_feat_y, feature_cols_y)

        print("\n" + "=" * 70)
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")

        print("\n[Step1] Training AutoGluon (Delta X)...")
        predictor_dx = fit_autogluon(
            x_train_feat_x,
            y_train_dx,
            x_valid_feat_x,
            y_valid_dx,
            "label_delta_x",
            f"{STEP1_DIR}/fold{fold_idx + 1}_delta_x",
        )
        pred_train_dx = predictor_dx.predict(x_train_feat_x[feature_cols_x]).values.astype(float)
        pred_valid_dx = predictor_dx.predict(x_valid_feat_x[feature_cols_x]).values.astype(float)

        print("\n[Step1] Training AutoGluon (Delta Y)...")
        predictor_dy = fit_autogluon(
            x_train_feat_y,
            y_train_dy,
            x_valid_feat_y,
            y_valid_dy,
            "label_delta_y",
            f"{STEP1_DIR}/fold{fold_idx + 1}_delta_y",
        )
        pred_train_dy = predictor_dy.predict(x_train_feat_y[feature_cols_y]).values.astype(float)
        pred_valid_dy = predictor_dy.predict(x_valid_feat_y[feature_cols_y]).values.astype(float)

        base_x = valid_start_x + pred_valid_dx
        base_y = valid_start_y + pred_valid_dy
        base_x, base_y = conservative_postprocess(base_x, base_y, x_valid_fold)
        base_dist = dist_mean(base_x, base_y, y_valid_abs_x, y_valid_abs_y)
        print(f"Base distance: {base_dist:.4f}")

        res_train_dx = (y_train_dx - pred_train_dx).astype("float32")
        res_train_dy = (y_train_dy - pred_train_dy).astype("float32")

        x_train_res_x = x_train_feat_x.copy()
        x_valid_res_x = x_valid_feat_x.copy()
        x_train_res_x["base_pred_dx"] = pred_train_dx
        x_valid_res_x["base_pred_dx"] = pred_valid_dx

        x_train_res_y = x_train_feat_y.copy()
        x_valid_res_y = x_valid_feat_y.copy()
        x_train_res_y["base_pred_dy"] = pred_train_dy
        x_valid_res_y["base_pred_dy"] = pred_valid_dy

        seed_fold = RANDOM_SEED + 1000 + fold_idx
        pack = tune_step2_separate_studies_for_fold(
            fold_idx,
            x_train_res_x, res_train_dx,
            x_train_res_y, res_train_dy,
            x_valid_res_x, pred_valid_dx,
            x_valid_res_y, pred_valid_dy,
            valid_start_x, valid_start_y,
            y_valid_abs_x, y_valid_abs_y,
            x_valid_fold,
            seed=seed_fold,
            trials_dx=OPTUNA_TRIALS_DX,
            trials_dy=OPTUNA_TRIALS_DY,
        )

        save_step2_pack(fold_idx, pack)

        feat_cols_x_step2 = pack["feat_cols_x"]
        feat_cols_y_step2 = pack["feat_cols_y"]

        x_valid_res_x_aligned = align_columns(x_valid_res_x, feat_cols_x_step2)
        x_valid_res_y_aligned = align_columns(x_valid_res_y, feat_cols_y_step2)

        rdx = pack["best_model_dx"].predict(catboost_pool(x_valid_res_x_aligned)).astype("float32")
        rdy = pack["best_model_dy"].predict(catboost_pool(x_valid_res_y_aligned)).astype("float32")

        ax_chk = pack["alpha_x"]
        ay_chk = pack["alpha_y"]

        chk_x = valid_start_x + (pred_valid_dx + ax_chk * rdx)
        chk_y = valid_start_y + (pred_valid_dy + ay_chk * rdy)
        chk_x, chk_y = conservative_postprocess(chk_x, chk_y, x_valid_fold)
        best_dist_check = dist_mean(chk_x, chk_y, y_valid_abs_x, y_valid_abs_y)

        print(f"[Step2] Check distance: {best_dist_check:.4f} (alpha_x={ax_chk:.3f}, alpha_y={ay_chk:.3f})")

        fold_results.append({
            "fold_idx": fold_idx,
            "base_distance": base_dist,
            "distance": float(best_dist_check),
            "alpha_x": float(ax_chk),
            "alpha_y": float(ay_chk),
        })

    fold_results_df = pd.DataFrame(fold_results).sort_values("distance")
    top_k = min(4, len(fold_results_df))
    top_folds = fold_results_df.head(top_k)["fold_idx"].values.astype(int).tolist()

    print("\n" + "=" * 70)
    print("ALL FOLD RESULTS")
    for _, r in fold_results_df.iterrows():
        mark = " *" if int(r["fold_idx"]) in top_folds else ""
        print(
            f"Fold {int(r['fold_idx']) + 1}: "
            f"base={r['base_distance']:.4f} -> step2={r['distance']:.4f} "
            f"(ax={r['alpha_x']:.3f}, ay={r['alpha_y']:.3f}){mark}"
        )
    print(f"Top{top_k} avg distance: {fold_results_df.head(top_k)['distance'].mean():.4f}")
    print("=" * 70)

    fold_results_df.to_csv(FOLD_RESULTS_PATH, index=False)

    meta = {
        "feature_cols_x": global_feature_cols_x,
        "feature_cols_y": global_feature_cols_y,
        "top_folds": top_folds,
    }
    with open(PIPELINE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved fold summary: {FOLD_RESULTS_PATH}")
    print(f"Saved pipeline meta: {PIPELINE_META_PATH}")

    return {
        "top_folds": top_folds,
        "feature_cols_x": global_feature_cols_x,
        "feature_cols_y": global_feature_cols_y,
    }


if __name__ == "__main__":
    train_pipeline()
