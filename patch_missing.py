"""
patch_missing.py
----------------
Re-pulls the 21 players that pull_data.py skipped due to playerid_lookup
failures (accents, periods, Jr. suffixes, common-name collisions).
Uses hardcoded MLBAM IDs sourced from Baseball Savant.
Appends the new rows to the existing barrel_data.parquet.
"""

import numpy as np
import pandas as pd
import pybaseball
from pybaseball import statcast_batter

pybaseball.cache.enable()

SEASONS = [2023, 2024, 2025]
OUT_PATH = "C:/Users/jaico/baseball/barrel_data.parquet"

SWING_EVENTS = {
    "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
}

# Hardcoded MLBAM IDs for the 21 skipped players
MISSING = {
    "Adolis Garcia":        666969,
    "CJ Cron":              543068,
    "Eloy Jimenez":         650391,
    "Eugenio Suarez":       553993,
    "Fernando Tatis Jr.":   665487,
    "J.D. Martinez":        502110,
    "Javier Baez":          595879,
    "Jeremy Pena":          665161,
    "Jose Abreu":           547989,
    "Jose Altuve":          514888,
    "Jose Ramirez":         608070,
    "JP Crawford":          641487,
    "Julio Rodriguez":      677594,
    "Luis Arraez":          650333,
    "Ramon Laureano":       598264,
    "Ronald Acuna Jr.":     660670,
    "Salvador Perez":       521692,
    "Teoscar Hernandez":    606192,
    "Victor Robles":        645302,
    "Yandy Diaz":           650490,
    "Yordan Alvarez":       670541,
}

def classify(df: pd.DataFrame, player_name: str, season: int) -> pd.DataFrame:
    df = df.copy()
    df["player_name_display"] = player_name
    df["season"]              = season
    df["is_swing"] = df["description"].isin(SWING_EVENTS)

    ev     = df["launch_speed"].to_numpy(dtype=float, na_value=np.nan)
    la     = df["launch_angle"].to_numpy(dtype=float, na_value=np.nan)
    ev_cap = np.clip(ev, 0, 116)
    delta  = ev_cap - 98
    min_la = 26 - delta
    max_la = 30 + delta * (20.0 / 18.0)

    df["is_barrel"] = (
        (~np.isnan(ev)) & (~np.isnan(la)) &
        (ev >= 98) &
        (la >= min_la) & (la <= max_la)
    )
    return df

total  = len(MISSING) * len(SEASONS)
done   = 0
chunks = []

print(f"Patching {len(MISSING)} players × {len(SEASONS)} seasons = {total} requests\n")

for player_name, mlbam_id in MISSING.items():
    for season in SEASONS:
        done += 1
        label = f"[{done:>2}/{total}]  {player_name} {season}"
        try:
            raw = statcast_batter(f"{season}-03-01", f"{season}-11-30", mlbam_id)
            if raw is None or raw.empty:
                print(f"  [empty]  {label}")
                continue
            df = classify(raw, player_name, season)
            chunks.append(df)
            print(f"  [ok]     {label}  — {len(df):,} pitches")
        except Exception as e:
            print(f"  [error]  {label}: {e}")

if not chunks:
    print("\nNothing new to append.")
else:
    print(f"\nLoading existing parquet …")
    existing = pd.read_parquet(OUT_PATH)
    print(f"  Existing rows : {len(existing):,}")

    new_data = pd.concat(chunks, ignore_index=True)
    for col in ["is_swing", "is_barrel"]:
        new_data[col] = new_data[col].astype(bool)
    new_data["season"] = new_data["season"].astype("int16")

    combined = pd.concat([existing, new_data], ignore_index=True)
    print(f"  New rows      : {len(new_data):,}")
    print(f"  Total rows    : {len(combined):,}")
    print(f"Saving to {OUT_PATH} …")
    combined.to_parquet(OUT_PATH, index=False, engine="pyarrow", compression="snappy")
    import os
    size_mb = os.path.getsize(OUT_PATH) / 1_048_576
    print(f"Done — {size_mb:.1f} MB written.")
