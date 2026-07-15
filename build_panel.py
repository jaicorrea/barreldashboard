import io

import pandas as pd
import requests

df = pd.read_parquet("barrel_data.parquet")
df = df[df["game_type"] == "R"]

# woba_denom is occasionally missing on PA-ending pitches; standard wOBA rule is
# denom=1 for all PA outcomes except sac bunts / catcher interference (denom=0)
zero_denom_events = {"sac_bunt", "catcher_interf"}
needs_fill = df["woba_value"].notna() & df["woba_denom"].isna()
df.loc[needs_fill, "woba_denom"] = (~df.loc[needs_fill, "events"].isin(zero_denom_events)).astype(int)

is_bbe = df["type"] == "X"  # batted ball event

agg = (
    df.groupby(["batter", "season"])
    .agg(
        player_name=("player_name", "first"),
        n_pitches=("pitch_type", "size"),
        n_swings=("is_swing", "sum"),
        n_bbe=("type", lambda s: (s == "X").sum()),
        n_barrels=("is_barrel", "sum"),
        avg_launch_speed=("launch_speed", "mean"),
        avg_launch_angle=("launch_angle", "mean"),
        avg_bat_speed=("bat_speed", "mean"),
        avg_swing_length=("swing_length", "mean"),
        woba_sum=("woba_value", "sum"),
        woba_denom=("woba_denom", "sum"),
        n_pa=("woba_value", "count"),
    )
    .reset_index()
)

agg["swing_pct"] = agg["n_swings"] / agg["n_pitches"]
agg["barrel_pct_bbe"] = agg["n_barrels"] / agg["n_bbe"]
agg["barrel_pct_swing"] = agg["n_barrels"] / agg["n_swings"]
agg = agg.drop(columns=["woba_sum", "woba_denom"])
agg = agg.rename(columns={"batter": "batter_id"})

# ── Join official wOBA from Baseball Savant leaderboard ───────────────────────
frames = []
for year in sorted(agg["season"].unique()):
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/expected_statistics"
        f"?type=batter&year={int(year)}&position=&team=&min=1&csv=true"
    )
    r = requests.get(url, timeout=30)
    tmp = pd.read_csv(io.StringIO(r.text))
    tmp["season"] = year
    frames.append(tmp[["player_id", "season", "woba"]])

savant = pd.concat(frames, ignore_index=True).rename(columns={"player_id": "batter_id", "woba": "woba"})
agg = agg.merge(savant, on=["batter_id", "season"], how="left")

agg = agg.sort_values(["batter_id", "season"]).reset_index(drop=True)

agg.to_csv("batter_season_panel.csv", index=False)
agg.to_parquet("batter_season_panel.parquet", index=False)

print(agg.shape)
print(agg.head(10))
