import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    TRAIN_PATH,
    TEST_META_PATH,
    SAMPLE_SUB_PATH,
    PIPELINE_META_PATH,
    SUBMISSION_PATH,
    USE_PLAYER_FEATURES,
    USE_TEAM_FEATURES,
)
from src.features import build_stable_features
from src.model_utils import load_step1_predictor, load_step2_pack, align_columns, catboost_pool
from src.stats import compute_player_statistics_simple, compute_team_statistics_simple
from src.utils import (
    configure_environment,
    add_norm_coords,
    apply_log_transform,
    cast_categoricals,
    replace_inf,
    conservative_postprocess,
)


def predict_test():
    configure_environment()

    print("=" * 80)
    print("Predict: load trained Step1/Step2 models and generate submission")
    print("=" * 80)

    with open(PIPELINE_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols_x = meta["feature_cols_x"]
    feature_cols_y = meta["feature_cols_y"]
    top_folds = meta["top_folds"]

    print("\nLoading data...")
    train_df_raw = pd.read_csv(TRAIN_PATH)
    test_meta = pd.read_csv(TEST_META_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

    print("\nLoading raw test episodes...")
    test_df_raw = pd.concat(
        [pd.read_csv(r["path"]).assign(game_episode=r["game_episode"]) for _, r in tqdm(test_meta.iterrows(), total=len(test_meta))],
        ignore_index=True,
    )

    player_stats_test = compute_player_statistics_simple(train_df_raw, 10) if USE_PLAYER_FEATURES else None
    team_stats_test = compute_team_statistics_simple(train_df_raw, 50) if USE_TEAM_FEATURES else None

    test_feats = build_stable_features(test_df_raw, player_stats_test, team_stats_test, False)
    test_feats = cast_categoricals(apply_log_transform(add_norm_coords(test_feats)))
    test_feats = test_feats.merge(test_meta[["game_episode"]], on="game_episode", how="right")

    x_test_feat_x = test_feats[feature_cols_x].copy()
    x_test_feat_y = test_feats[feature_cols_y].copy()
    replace_inf(x_test_feat_x, feature_cols_x)
    replace_inf(x_test_feat_y, feature_cols_y)

    test_start_x = test_feats["start_x_last"].values.astype(float)
    test_start_y = test_feats["start_y_last"].values.astype(float)

    pred_x_list, pred_y_list = [], []

    for fold_idx in top_folds:
        predictor_dx = load_step1_predictor(fold_idx, "x")
        predictor_dy = load_step1_predictor(fold_idx, "y")

        pdx = predictor_dx.predict(x_test_feat_x[feature_cols_x]).values.astype(float)
        pdy = predictor_dy.predict(x_test_feat_y[feature_cols_y]).values.astype(float)

        pack_loaded = load_step2_pack(fold_idx)
        m_dx, m_dy = pack_loaded["m_dx"], pack_loaded["m_dy"]
        ax, ay = pack_loaded["alpha_x"], pack_loaded["alpha_y"]
        feat_cols_x_step2 = pack_loaded["feat_cols_x"]
        feat_cols_y_step2 = pack_loaded["feat_cols_y"]

        x_test_res_x = x_test_feat_x.copy()
        x_test_res_y = x_test_feat_y.copy()
        x_test_res_x["base_pred_dx"] = pdx
        x_test_res_y["base_pred_dy"] = pdy

        x_test_res_x = align_columns(x_test_res_x, feat_cols_x_step2)
        x_test_res_y = align_columns(x_test_res_y, feat_cols_y_step2)

        rdx = m_dx.predict(catboost_pool(x_test_res_x)).astype("float32")
        rdy = m_dy.predict(catboost_pool(x_test_res_y)).astype("float32")

        pred_x_list.append(test_start_x + (pdx + ax * rdx))
        pred_y_list.append(test_start_y + (pdy + ay * rdy))

    test_pred_x = np.mean(np.vstack(pred_x_list), axis=0)
    test_pred_y = np.mean(np.vstack(pred_y_list), axis=0)
    test_pred_x, test_pred_y = conservative_postprocess(test_pred_x, test_pred_y, test_feats)

    submission = sample_sub.copy().merge(test_meta[["game_episode"]], on="game_episode", how="left")
    submission["end_x"] = test_pred_x
    submission["end_y"] = test_pred_y

    submission[["game_episode", "end_x", "end_y"]].to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved: {SUBMISSION_PATH}")


if __name__ == "__main__":
    predict_test()
