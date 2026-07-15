import pandas as pd
import pybaseball as pb

pb.cache.enable()

panel = pd.read_csv("batter_season_panel.csv")

truth_frames = []
for season in sorted(panel["season"].unique()):
    t = pb.batting_stats_bref(int(season))[["mlbID", "PA"]].copy()
    t["mlbID"] = pd.to_numeric(t["mlbID"], errors="coerce")
    t = t.dropna(subset=["mlbID"])
    t["mlbID"] = t["mlbID"].astype(int)
    # traded players have a per-stint row + a combined "TOT" row from bbref;
    # the combined total is always the max PA for that player-season
    t = t.groupby("mlbID", as_index=False)["PA"].max()
    t["season"] = season
    truth_frames.append(t)

truth = pd.concat(truth_frames, ignore_index=True)
truth = truth.rename(columns={"mlbID": "batter_id", "PA": "true_pa"})

merged = panel.merge(truth, on=["batter_id", "season"], how="left")
merged["pa_gap"] = merged["true_pa"] - merged["n_pa"]
merged["pa_gap_pct"] = merged["pa_gap"] / merged["true_pa"]

merged.to_csv("pa_coverage_check.csv", index=False)

no_match = merged[merged["true_pa"].isna()]
flagged = merged[merged["pa_gap_pct"].abs() > 0.05].sort_values("pa_gap_pct", ascending=False)

print(f"Total batter-seasons: {len(merged)}")
print(f"No bbref match found: {len(no_match)}")
if len(no_match):
    print(no_match[["batter_id", "player_name", "season"]].to_string(index=False))
print(f"\nFlagged (>5% PA gap vs true): {len(flagged)}")
print(flagged[["batter_id", "player_name", "season", "n_pa", "true_pa", "pa_gap", "pa_gap_pct"]].to_string(index=False))
