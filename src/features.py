import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    MAX_EPISODE_EVENTS,
    GOAL_X,
    GOAL_Y,
    ABLATION_CONFIG,
    USE_PLAYER_FEATURES,
    USE_TEAM_FEATURES,
    N_FOLDS,
    VAL_GAME_RATIO,
    FEATURES_TO_REMOVE_X,
    FEATURES_TO_REMOVE_Y,
)
from src.utils import (
    merge_stats_fillna,
    add_norm_coords,
    apply_log_transform,
    clip_outliers,
    cast_categoricals,
)


def build_stable_features(df: pd.DataFrame, player_stats_df=None, team_stats_df=None, is_train=True):
    df = df.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True)
    feats, yx, yy = [], [], []

    def time_since_last_type(hist, last_time, type_name):
        if len(hist) == 0:
            return 999.0
        sub = hist[hist["type_name"] == type_name]
        if len(sub) == 0:
            return 999.0
        return float(last_time - sub["time_seconds"].iloc[-1])

    for ep, g in tqdm(df.groupby("game_episode"), desc="Building features"):
        g = g.reset_index(drop=True)
        if len(g) > MAX_EPISODE_EVENTS:
            g = g.tail(MAX_EPISODE_EVENTS).reset_index(drop=True)

        n = len(g)
        if n == 0:
            continue

        last = g.iloc[-1]
        hist = g.iloc[:-1] if n > 1 else g.iloc[0:0]

        feat = {
            "game_episode": ep,
            "game_id": int(last["game_id"]),
            "period_id": int(last["period_id"]),
            "episode_id": int(last["episode_id"]),
            "time_seconds_last": float(last["time_seconds"]),
            "team_id_last": int(last["team_id"]),
            "player_id_last": int(last["player_id"]),
            "action_id_last": int(last["action_id"]),
            "type_name_last": str(last["type_name"]),
            "result_name_last": "NoResult" if pd.isna(last["result_name"]) else str(last["result_name"]),
            "start_x_last": float(last["start_x"]),
            "start_y_last": float(last["start_y"]),
            "is_home_last": int(bool(last["is_home"])),
            "n_events": n,
            "n_hist_events": len(hist),
        }

        t_start, t_end = float(g["time_seconds"].iloc[0]), float(last["time_seconds"])
        dur = max(t_end - t_start, 1e-6)
        feat["dur_episode"] = dur
        feat["events_per_10s"] = (len(hist) / dur) * 10.0 if len(hist) > 0 else 0.0

        period_duration = 45 * 60
        feat["rel_time_in_period"] = float(np.clip(feat["time_seconds_last"] / period_duration, 0.0, 2.0))

        if len(hist) > 0:
            prev_event = hist.iloc[-1]
            feat.update({
                "prev_is_carry": int(prev_event["type_name"] == "Carry"),
                "prev_is_catch": int(prev_event["type_name"] == "Catch"),
                "prev_is_recovery": int(prev_event["type_name"] == "Recovery"),
                "prev_is_duel": int(prev_event["type_name"] == "Duel"),
            })
        else:
            feat.update({"prev_is_carry": 0, "prev_is_catch": 0, "prev_is_recovery": 0, "prev_is_duel": 0})

        first_type = g.iloc[0]["type_name"]
        feat.update({
            "started_from_throwin": int(first_type == "Throw-In"),
            "started_from_goalkick": int(first_type == "Goal Kick"),
            "started_from_corner": int(first_type == "Pass_Corner"),
            "started_from_freekick": int(first_type == "Pass_Freekick"),
        })

        num_cols = ["start_x", "start_y", "end_x", "end_y"]
        if len(hist) > 0:
            for col in num_cols:
                feat[f"{col}_mean_hist"] = float(hist[col].mean())
                feat[f"{col}_std_hist"] = float(hist[col].std(ddof=0)) if len(hist) > 1 else 0.0
                feat[f"{col}_min_hist"] = float(hist[col].min())
                feat[f"{col}_max_hist"] = float(hist[col].max())
        else:
            for col in num_cols:
                feat[f"{col}_mean_hist"] = 0.0
                feat[f"{col}_std_hist"] = 0.0
                feat[f"{col}_min_hist"] = 0.0
                feat[f"{col}_max_hist"] = 0.0

        if len(hist) > 0:
            dx_hist = hist["end_x"].values - hist["start_x"].values
            dy_hist = hist["end_y"].values - hist["start_y"].values
            step_dists = np.sqrt(dx_hist**2 + dy_hist**2)
            feat["path_len_hist"] = float(step_dists.sum())
            feat["mean_step_dist_hist"] = float(step_dists.mean())
            step_angles = np.degrees(np.arctan2(dy_hist, dx_hist))
            feat["mean_step_angle_hist"] = float(step_angles.mean())
            feat["std_step_angle_hist"] = float(step_angles.std(ddof=0)) if len(step_angles) > 1 else 0.0
            feat["last_step_angle_hist"] = float(step_angles[-1])
        else:
            dx_hist = np.array([], dtype=float)
            dy_hist = np.array([], dtype=float)
            step_dists = np.array([], dtype=float)
            feat.update({
                "path_len_hist": 0.0,
                "mean_step_dist_hist": 0.0,
                "mean_step_angle_hist": 0.0,
                "std_step_angle_hist": 0.0,
                "last_step_angle_hist": 0.0,
            })

        feat["mean_speed_hist"] = feat["path_len_hist"] / dur if dur > 0 else 0.0

        first_start_x, first_start_y = float(g["start_x"].iloc[0]), float(g["start_y"].iloc[0])
        feat["start_x_first"] = first_start_x
        feat["start_y_first"] = first_start_y
        feat["delta_start_x_first_last"] = feat["start_x_last"] - first_start_x
        feat["delta_start_y_first_last"] = feat["start_y_last"] - first_start_y

        if len(hist) > 0:
            prev_end_x, prev_end_y = float(hist["end_x"].iloc[-1]), float(hist["end_y"].iloc[-1])
            prev_time = float(hist["time_seconds"].iloc[-1])
        else:
            prev_end_x = 0.0
            prev_end_y = 0.0
            prev_time = float(last["time_seconds"])

        feat["prev_end_x"] = prev_end_x
        feat["prev_end_y"] = prev_end_y
        feat["delta_prev_end_to_last_start_x"] = feat["start_x_last"] - prev_end_x
        feat["delta_prev_end_to_last_start_y"] = feat["start_y_last"] - prev_end_y

        dist_prev = np.sqrt(feat["delta_prev_end_to_last_start_x"]**2 + feat["delta_prev_end_to_last_start_y"]**2)
        dt_prev = max(feat["time_seconds_last"] - prev_time, 1e-6)
        feat["dist_prev_to_last_start"] = float(dist_prev)
        feat["speed_prev_to_last_start"] = float(dist_prev / dt_prev)
        feat["angle_prev_to_last_start"] = float(
            np.degrees(np.arctan2(feat["delta_prev_end_to_last_start_y"], feat["delta_prev_end_to_last_start_x"]))
        )

        for k in range(1, 4):
            if len(hist) >= k:
                row = hist.iloc[-k]
                feat[f"lag{k}_start_x"] = float(row["start_x"])
                feat[f"lag{k}_start_y"] = float(row["start_y"])
                feat[f"lag{k}_end_x"] = float(row["end_x"])
                feat[f"lag{k}_end_y"] = float(row["end_y"])
                feat[f"lag{k}_time_seconds"] = float(row["time_seconds"])
                dx_lag = feat["start_x_last"] - feat[f"lag{k}_start_x"]
                dy_lag = feat["start_y_last"] - feat[f"lag{k}_start_y"]
                feat[f"lag{k}_dist_to_curr"] = float(np.sqrt(dx_lag**2 + dy_lag**2))
                feat[f"lag{k}_angle_to_curr"] = float(np.degrees(np.arctan2(dy_lag, dx_lag)))
            else:
                for s in ["start_x", "start_y", "end_x", "end_y", "time_seconds", "dist_to_curr", "angle_to_curr"]:
                    feat[f"lag{k}_{s}"] = 0.0

        if len(hist) > 0:
            alpha = 0.3
            for col in num_cols:
                feat[f"{col}_ewm_hist"] = float(hist[col].ewm(alpha=alpha, adjust=False).mean().iloc[-1])
            feat["step_dist_ewm_hist"] = (
                float(pd.Series(step_dists).ewm(alpha=alpha, adjust=False).mean().iloc[-1])
                if len(step_dists) > 0 else 0.0
            )
        else:
            for col in num_cols:
                feat[f"{col}_ewm_hist"] = 0.0
            feat["step_dist_ewm_hist"] = 0.0

        dx_goal, dy_goal = GOAL_X - feat["start_x_last"], GOAL_Y - feat["start_y_last"]
        feat["dist_to_goal_from_start_last"] = float(np.sqrt(dx_goal**2 + dy_goal**2))
        feat["angle_to_goal_from_start_last"] = float(np.degrees(np.arctan2(dy_goal, dx_goal)))

        dx_goal_prev, dy_goal_prev = GOAL_X - prev_end_x, GOAL_Y - prev_end_y
        feat["dist_to_goal_from_prev_end"] = float(np.sqrt(dx_goal_prev**2 + dy_goal_prev**2))
        feat["delta_dist_goal_prev_to_last"] = feat["dist_to_goal_from_prev_end"] - feat["dist_to_goal_from_start_last"]

        if len(hist) > 0:
            team_vals = hist["team_id"].values
            time_vals = hist["time_seconds"].values
            feat["team_switch_count_hist"] = int(np.sum(team_vals[1:] != team_vals[:-1]) if len(team_vals) > 1 else 0)
            feat["home_possession_ratio_hist"] = float(hist["is_home"].mean())
            prev_team = int(hist["team_id"].iloc[-1])
            feat["is_same_team_as_prev"] = int(prev_team == feat["team_id_last"])

            last_team = feat["team_id_last"]
            poss_len = 0
            for i in range(len(team_vals) - 1, -1, -1):
                if team_vals[i] == last_team:
                    poss_len += 1
                else:
                    break
            feat["current_team_possession_len"] = poss_len
            if poss_len > 0:
                poss_start_idx = len(team_vals) - poss_len
                poss_start_time = float(time_vals[poss_start_idx])
                feat["current_team_possession_time"] = float(feat["time_seconds_last"] - poss_start_time)
            else:
                feat["current_team_possession_time"] = 0.0
        else:
            feat.update({
                "team_switch_count_hist": 0,
                "home_possession_ratio_hist": 0.0,
                "is_same_team_as_prev": 0,
                "current_team_possession_len": 0,
                "current_team_possession_time": 0.0,
            })

        type_counts = hist["type_name"].value_counts() if len(hist) > 0 else pd.Series(dtype=int)
        key_types = ["Pass", "Carry", "Shot", "Cross", "Duel", "Interception", "Tackle", "Clearance", "Recovery"]
        for t in key_types:
            feat[f"cnt_{t}"] = int(type_counts.get(t, 0))

        denom = feat["n_hist_events"]
        feat["pass_ratio"] = feat["cnt_Pass"] / denom if denom > 0 else 0.0
        feat["shot_ratio"] = feat["cnt_Shot"] / denom if denom > 0 else 0.0
        feat["cross_ratio"] = feat["cnt_Cross"] / denom if denom > 0 else 0.0
        feat["duel_ratio"] = feat["cnt_Duel"] / denom if denom > 0 else 0.0

        type_last = feat["type_name_last"]
        feat.update({
            "is_shot_last": int(type_last == "Shot"),
            "is_pass_last": int(type_last == "Pass"),
            "is_cross_last": int(type_last == "Cross"),
            "is_carry_last": int(type_last == "Carry"),
        })

        if len(hist) > 0:
            forward_mask = dx_hist > 0
            feat["forward_ratio_hist"] = float(forward_mask.mean()) if len(dx_hist) > 0 else 0.0
            feat["mean_abs_lateral_hist"] = float(np.abs(dy_hist).mean()) if len(dy_hist) > 0 else 0.0
            feat["std_forward_hist"] = float(dx_hist.std(ddof=0)) if len(dx_hist) > 1 else 0.0
            feat["std_lateral_hist"] = float(dy_hist.std(ddof=0)) if len(dy_hist) > 1 else 0.0
        else:
            feat.update({
                "forward_ratio_hist": 0.0,
                "mean_abs_lateral_hist": 0.0,
                "std_forward_hist": 0.0,
                "std_lateral_hist": 0.0,
            })

        feat["cumulative_forward_progress"] = (
            float((hist["end_x"] - hist["start_x"]).cumsum().iloc[-1]) if len(hist) > 0 else 0.0
        )

        if len(hist) >= 2:
            recent = hist.tail(5)
            forward_progress = (recent["end_x"] - recent["start_x"]).sum()
            time_diff = recent["time_seconds"].max() - recent["time_seconds"].min()
            feat["attack_tempo"] = float(forward_progress / time_diff) if time_diff > 0 else 0.0
        else:
            feat["attack_tempo"] = 0.0

        if len(hist) >= 3:
            recent_3 = hist.tail(3)
            dx1 = recent_3["end_x"].iloc[-2] - recent_3["start_x"].iloc[-2]
            dt1 = recent_3["time_seconds"].iloc[-2] - recent_3["time_seconds"].iloc[-3] + 1e-6
            dx2 = recent_3["end_x"].iloc[-1] - recent_3["start_x"].iloc[-1]
            dt2 = recent_3["time_seconds"].iloc[-1] - recent_3["time_seconds"].iloc[-2] + 1e-6
            feat["recent_forward_accel"] = float(dx2 / dt2 - dx1 / dt1)
        else:
            feat["recent_forward_accel"] = 0.0

        if len(hist) >= 3:
            recent_steps = hist.tail(3)
            vx1 = recent_steps["end_x"].iloc[-2] - recent_steps["start_x"].iloc[-2]
            vy1 = recent_steps["end_y"].iloc[-2] - recent_steps["start_y"].iloc[-2]
            vx2 = recent_steps["end_x"].iloc[-1] - recent_steps["start_x"].iloc[-1]
            vy2 = recent_steps["end_y"].iloc[-1] - recent_steps["start_y"].iloc[-1]
            dot = vx1 * vx2 + vy1 * vy2
            norm1 = np.sqrt(vx1**2 + vy1**2) + 1e-6
            norm2 = np.sqrt(vx2**2 + vy2**2) + 1e-6
            cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
            feat["recent_direction_change_deg"] = float(np.degrees(np.arccos(cos_angle)))
        else:
            feat["recent_direction_change_deg"] = 0.0

        feat["open_space_forward"] = max(0.0, GOAL_X - feat["start_x_last"])
        feat["open_space_left"] = feat["start_y_last"]
        feat["open_space_right"] = 68.0 - feat["start_y_last"]

        x, y = feat["start_x_last"], feat["start_y_last"]
        feat["x_zone"] = 0 if x < 35 else (1 if x < 70 else 2)
        feat["y_zone"] = 0 if y < 22.67 else (1 if y < 45.33 else 2)

        feat["is_danger_zone"] = int(x > 70 and 20 < y < 48)
        feat["is_wing"] = int(y < 10 or y > 58)
        feat["is_box"] = int(x > 88.5 and 13.84 < y < 54.16)

        if len(hist) > 0:
            wing_hist = hist[(hist["start_y"] < 10) | (hist["start_y"] > 58)]
            feat["wing_cross_ratio_hist"] = float((wing_hist["type_name"] == "Cross").mean()) if len(wing_hist) > 0 else 0.0
            box_hist = hist[(hist["start_x"] > 88.5) & (hist["start_y"] > 13.84) & (hist["start_y"] < 54.16)]
            feat["box_shot_ratio_hist"] = float((box_hist["type_name"] == "Shot").mean()) if len(box_hist) > 0 else 0.0
        else:
            feat["wing_cross_ratio_hist"] = 0.0
            feat["box_shot_ratio_hist"] = 0.0

        if len(hist) >= 5:
            recent_5 = hist.tail(5)
            feat["recent5_mean_forward"] = float((recent_5["end_x"] - recent_5["start_x"]).mean())
            feat["recent5_mean_lateral"] = float((recent_5["end_y"] - recent_5["start_y"]).mean())
            feat["recent5_std_forward"] = float((recent_5["end_x"] - recent_5["start_x"]).std())
            feat["recent5_std_lateral"] = float((recent_5["end_y"] - recent_5["start_y"]).std())
        else:
            feat.update({
                "recent5_mean_forward": 0.0,
                "recent5_mean_lateral": 0.0,
                "recent5_std_forward": 0.0,
                "recent5_std_lateral": 0.0,
            })

        if len(hist) > 0:
            recent_k = hist.tail(5)
            feat["recent_pass_ratio_k"] = float((recent_k["type_name"] == "Pass").mean())
            feat["recent_shot_ratio_k"] = float((recent_k["type_name"] == "Shot").mean())
            feat["recent_cross_ratio_k"] = float((recent_k["type_name"] == "Cross").mean())
        else:
            feat["recent_pass_ratio_k"] = 0.0
            feat["recent_shot_ratio_k"] = 0.0
            feat["recent_cross_ratio_k"] = 0.0

        feat["episode_progress"] = min(feat["n_events"] / 100.0, 1.0)

        angle_rad = np.radians(feat["angle_to_goal_from_start_last"])
        feat["goal_angle_sin"] = float(np.sin(angle_rad))
        feat["goal_angle_cos"] = float(np.cos(angle_rad))

        if len(hist) > 0:
            is_pass = (hist["type_name"] == "Pass").values
            pass_chain = 0
            for i in range(len(is_pass) - 1, -1, -1):
                if is_pass[i]:
                    pass_chain += 1
                else:
                    break
            feat["pass_chain_length"] = pass_chain
        else:
            feat["pass_chain_length"] = 0

        feat["is_counter_attack"] = int((len(hist) >= 3) and (dur < 10) and (feat["cumulative_forward_progress"] > 20))
        feat["y_bias"] = float(hist["end_y"].mean() - 34.0) if len(hist) > 0 else 0.0

        if len(hist) >= 3:
            recent_3 = hist.tail(3)
            feat["recent3_mean_x"] = float(recent_3["end_x"].mean())
            feat["recent3_mean_y"] = float(recent_3["end_y"].mean())
        else:
            feat["recent3_mean_x"] = feat["start_x_last"]
            feat["recent3_mean_y"] = feat["start_y_last"]

        feat["xy_interaction"] = feat["start_x_last"] * feat["start_y_last"] / 1000.0

        dist_goal = feat["dist_to_goal_from_start_last"]
        feat["goal_dist_zone"] = 0 if dist_goal < 20 else (1 if dist_goal < 40 else (2 if dist_goal < 70 else 3))
        feat["has_enough_hist"] = int(feat["n_hist_events"] >= 5)

        if ABLATION_CONFIG["USE_TS3_TIME_SINCE_EVENT"]:
            feat["time_since_last_shot"] = time_since_last_type(hist, feat["time_seconds_last"], "Shot")
            feat["time_since_last_cross"] = time_since_last_type(hist, feat["time_seconds_last"], "Cross")
            feat["time_since_last_pass"] = time_since_last_type(hist, feat["time_seconds_last"], "Pass")
        else:
            feat["time_since_last_shot"] = 999.0
            feat["time_since_last_cross"] = 999.0
            feat["time_since_last_pass"] = 999.0

        if is_train:
            yx.append(float(last["end_x"]))
            yy.append(float(last["end_y"]))

        feats.append(feat)

    features_df = pd.DataFrame(feats)
    if USE_PLAYER_FEATURES and player_stats_df is not None:
        features_df = merge_stats_fillna(features_df, player_stats_df, "player_id_last", "player_id", "player_id")
    if USE_TEAM_FEATURES and team_stats_df is not None:
        features_df = merge_stats_fillna(features_df, team_stats_df, "team_id_last", "team_id", "team_id")

    if is_train:
        return features_df, np.array(yx, dtype="float32"), np.array(yy, dtype="float32")
    return features_df


def setup_game_date_cv(train_df_raw: pd.DataFrame, match_info: pd.DataFrame):
    episode_meta_temp = train_df_raw.groupby("game_episode")["game_id"].first().reset_index()
    episode_meta_temp = episode_meta_temp.merge(match_info[["game_id", "game_date"]], on="game_id", how="left")
    game_date_df = episode_meta_temp[["game_id", "game_date"]].drop_duplicates().sort_values("game_date")
    sorted_games = game_date_df["game_id"].values
    n_games = len(sorted_games)

    total_val_games = max(N_FOLDS, int(n_games * VAL_GAME_RATIO))
    total_val_games = min(total_val_games, n_games - 1)
    val_games_per_fold = max(1, total_val_games // N_FOLDS)
    total_val_games = val_games_per_fold * N_FOLDS
    start_val_region = n_games - total_val_games
    return sorted_games, val_games_per_fold, start_val_region


def fold_game_ids(sorted_games, val_games_per_fold, start_val_region, fold_idx):
    s = start_val_region + fold_idx * val_games_per_fold
    e = s + val_games_per_fold
    return set(sorted_games[s:e])


def build_train_base(train_df_raw: pd.DataFrame):
    train_feats_base, y_end_x, y_end_y = build_stable_features(train_df_raw, None, None, True)
    train_feats_base["label_end_x_abs"] = y_end_x
    train_feats_base["label_end_y_abs"] = y_end_y

    y_delta_x = y_end_x - train_feats_base["start_x_last"].values
    y_delta_y = y_end_y - train_feats_base["start_y_last"].values

    train_feats_base = add_norm_coords(train_feats_base)
    train_feats_base = apply_log_transform(train_feats_base)

    clip_cols = [c for c in train_feats_base.columns if any(k in c for k in ["speed", "dist", "path_len", "attack_tempo"])]
    train_feats_base = clip_outliers(train_feats_base, cols=clip_cols)
    train_feats_base = cast_categoricals(train_feats_base)

    x_all_base = train_feats_base.copy()
    x_all_base["label_delta_x"] = y_delta_x
    x_all_base["label_delta_y"] = y_delta_y
    return x_all_base


def split_xy(x_all: pd.DataFrame):
    drop_cols = ["label_delta_x", "label_delta_y", "label_end_x_abs", "label_end_y_abs", "game_episode"]
    feature_cols_all = [c for c in x_all.columns if c not in drop_cols]
    feature_cols_x = [f for f in feature_cols_all if f not in FEATURES_TO_REMOVE_X]
    feature_cols_y = [f for f in feature_cols_all if f not in FEATURES_TO_REMOVE_Y]
    return feature_cols_x, feature_cols_y
