import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_player_statistics_simple(df: pd.DataFrame, min_samples: int = 10):
    df = df.copy()
    df["delta_x"] = df["end_x"] - df["start_x"]
    df["delta_y"] = df["end_y"] - df["start_y"]
    df["move_dist"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)

    global_avg_dist = df["move_dist"].mean()
    global_forward_ratio = (df["delta_x"] > 0).mean()
    global_main_x = df["start_x"].mean()
    global_longpass_ratio = (df["move_dist"] > 20).mean()

    rows = {}
    for pid, g in tqdm(df.groupby("player_id"), desc="Player stats"):
        n = len(g)
        w = min(n / min_samples, 1.0)
        rows[pid] = {
            "player_avg_move_dist": w * g["move_dist"].mean() + (1 - w) * global_avg_dist,
            "player_forward_ratio": w * (g["delta_x"] > 0).mean() + (1 - w) * global_forward_ratio,
            "player_main_x_position": w * g["start_x"].mean() + (1 - w) * global_main_x,
            "player_longpass_ratio": w * (g["move_dist"] > 20).mean() + (1 - w) * global_longpass_ratio,
            "player_activity_log": np.log1p(n),
        }
    return pd.DataFrame(rows).T.reset_index().rename(columns={"index": "player_id"})


def compute_team_statistics_simple(df: pd.DataFrame, min_samples: int = 50):
    df = df.copy()
    df["delta_x"] = df["end_x"] - df["start_x"]
    df["delta_y"] = df["end_y"] - df["start_y"]
    df["move_dist"] = np.sqrt(df["delta_x"] ** 2 + df["delta_y"] ** 2)

    global_avg_dist = df["move_dist"].mean()
    global_forward_ratio = (df["delta_x"] > 0).mean()
    global_attack_zone = (df["start_x"] > 70).mean()
    global_cross_ratio = (df["type_name"] == "Cross").mean()
    global_shot_ratio = (df["type_name"] == "Shot").mean()

    rows = {}
    for tid, g in tqdm(df.groupby("team_id"), desc="Team stats"):
        n = len(g)
        w = min(n / min_samples, 1.0)
        poss_style = g.groupby("game_episode").size().mean() if "game_episode" in g.columns else 5.0
        rows[tid] = {
            "team_avg_pass_dist": w * g["move_dist"].mean() + (1 - w) * global_avg_dist,
            "team_forward_aggression": w * (g["delta_x"] > 0).mean() + (1 - w) * global_forward_ratio,
            "team_attack_zone_ratio": w * (g["start_x"] > 70).mean() + (1 - w) * global_attack_zone,
            "team_cross_tendency": w * (g["type_name"] == "Cross").mean() + (1 - w) * global_cross_ratio,
            "team_shot_frequency": w * (g["type_name"] == "Shot").mean() + (1 - w) * global_shot_ratio,
            "team_possession_style": poss_style,
            "team_activity_log": np.log1p(n),
        }
    return pd.DataFrame(rows).T.reset_index().rename(columns={"index": "team_id"})
