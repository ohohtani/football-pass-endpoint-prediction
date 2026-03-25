import os

# ============================================================
# Global config
# ============================================================
MAX_EPISODE_EVENTS = 20

TRAIN_PATH = "./data/train.csv"
TEST_META_PATH = "./data/test.csv"
MATCH_INFO_PATH = "./data/match_info.csv"
SAMPLE_SUB_PATH = "./data/sample_submission.csv"

RANDOM_SEED = 42
VAL_GAME_RATIO = 0.2
N_FOLDS = 5
USE_GPU = False

GOAL_X, GOAL_Y = 105.0, 34.0

ABLATION_CONFIG = {"USE_TS3_TIME_SINCE_EVENT": True}
RUN_FOLDS = [1, 3, 4]  # 0-indexed

USE_PLAYER_FEATURES = True
USE_TEAM_FEATURES = True

POSTPROC_LONGPASS_BOOST = True
LONGPASS_BOOST_GOAL_DIST_ZONE = 3
LONGPASS_BOOST_PRED_DIST_THRESH = 25.0
LONGPASS_BOOST_SCALE = 1.08

FEATURES_TO_REMOVE_COMMON = ['home_possession_ratio_hist', 'cnt_Tackle', 'game_id']

NEG_IMP_X_FEATURES = [
    "end_x_min_hist", "end_x_max_hist", "action_id_last", "box_shot_ratio_hist",
    "recent_pass_ratio_k", "cross_ratio", "start_x_mean_hist", "end_y_mean_hist",
    "y_bias", "is_home_last", "end_y_std_hist", "lag3_start_y", "cnt_Clearance",
    "recent_shot_ratio_k", "start_y_max_hist", "recent3_mean_x_norm",
    "has_enough_hist", "cnt_Recovery_log1p", "cnt_Cross", "rel_time_in_period",
    "cnt_Carry_log1p", "wing_cross_ratio_hist", "cnt_Shot_log1p", "shot_ratio",
    "recent_cross_ratio_k", "lag1_time_seconds", "period_id", "n_events_log1p",
    "is_same_team_as_prev", "n_hist_events", "episode_progress", "episode_id",
    "is_danger_zone", "goal_angle_sin", "time_seconds_last", "end_x_mean_hist",
    "team_id_last",
]

NEG_IMP_Y_FEATURES = [
    "start_x_first", "end_x_max_hist", "action_id_last", "cnt_Carry",
    "std_forward_hist", "start_x_last_norm", "shot_ratio", "cnt_Recovery",
    "recent_shot_ratio_k", "cnt_Cross_log1p", "has_enough_hist",
    "cnt_Interception", "cnt_Shot_log1p", "period_id", "cnt_Shot",
    "box_shot_ratio_hist", "time_since_last_shot", "recent_cross_ratio_k",
    "y_bias", "n_events_log1p", "n_hist_events", "cnt_Interception_log1p",
    "is_counter_attack", "end_y_max_hist", "start_x_std_hist", "cnt_Clearance",
    "current_team_possession_len", "time_since_last_cross",
    "team_switch_count_hist", "start_x_mean_hist", "n_hist_events_log1p",
    "lag3_start_x", "start_x_first_norm",
]

FEATURES_TO_REMOVE_X = sorted(set(FEATURES_TO_REMOVE_COMMON + NEG_IMP_X_FEATURES))
FEATURES_TO_REMOVE_Y = sorted(set(FEATURES_TO_REMOVE_COMMON + NEG_IMP_Y_FEATURES))

N_CPUS = 12
AUTOGLOUON_HYPERPARAMS = {
    "GBM": {"num_threads": N_CPUS},
    "CAT": {"thread_count": N_CPUS},
    "XGB": {"nthread": N_CPUS},
    "RF":  {"n_jobs": N_CPUS},
    "XT":  {"n_jobs": N_CPUS},
}

OPTUNA_TRIALS_DX = 20
OPTUNA_TRIALS_DY = 20

BASE_OUTPUT_DIR = "./outputs"
STEP1_DIR = os.path.join(BASE_OUTPUT_DIR, "model_split_2")
STEP2_DIR = os.path.join(BASE_OUTPUT_DIR, "model_2")
PIPELINE_META_PATH = os.path.join(BASE_OUTPUT_DIR, "pipeline_meta.json")
FOLD_RESULTS_PATH = os.path.join(BASE_OUTPUT_DIR, "fold_results.csv")
SUBMISSION_PATH = "./submission.csv"
