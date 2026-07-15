"""
pull_data.py
------------
Pulls per-pitch Statcast data for every player in the dashboard roster
for the past three seasons (2023-2025), classifies each pitch with the
same barrel / swing logic used in dashboard.py, and writes the result to
barrel_data.parquet in this folder.

Run:  python pull_data.py
"""

import time

import numpy as np
import pandas as pd
import pybaseball
from pybaseball import statcast_batter, playerid_lookup

pybaseball.cache.enable()

MAX_RETRIES = 3

# ── Seasons ────────────────────────────────────────────────────────────────────
SEASONS = [2023, 2024, 2025]

# ── Full roster (mirrors dashboard.py) ────────────────────────────────────────
PLAYERS = {
    "Aaron Judge":              {"first": "Aaron",       "last": "Judge"},
    "Adam Duvall":              {"first": "Adam",        "last": "Duvall"},
    "Adam Frazier":             {"first": "Adam",        "last": "Frazier"},
    "Adley Rutschman":          {"first": "Adley",       "last": "Rutschman"},
    "Adolis Garcia":            {"first": "Adolis",      "last": "Garcia",    "mlbam_id": 666969},
    "Alejandro Kirk":           {"first": "Alejandro",   "last": "Kirk"},
    "Alec Bohm":                {"first": "Alec",        "last": "Bohm"},
    "Alex Bregman":             {"first": "Alex",        "last": "Bregman"},
    "Alex Verdugo":             {"first": "Alex",        "last": "Verdugo"},
    "Andrew Benintendi":        {"first": "Andrew",      "last": "Benintendi"},
    "Andrew Vaughn":            {"first": "Andrew",      "last": "Vaughn"},
    "Anthony Rizzo":            {"first": "Anthony",     "last": "Rizzo"},
    "Anthony Santander":        {"first": "Anthony",     "last": "Santander"},
    "Anthony Volpe":            {"first": "Anthony",     "last": "Volpe"},
    "Austin Hays":              {"first": "Austin",      "last": "Hays"},
    "Austin Riley":             {"first": "Austin",      "last": "Riley"},
    "Bobby Witt Jr.":           {"first": "Bobby",       "last": "Witt"},
    "Bo Bichette":              {"first": "Bo",          "last": "Bichette"},
    "Brendan Rodgers":          {"first": "Brendan",     "last": "Rodgers"},
    "Bryce Harper":             {"first": "Bryce",       "last": "Harper"},
    "Bryan Reynolds":           {"first": "Bryan",       "last": "Reynolds"},
    "Cal Raleigh":              {"first": "Cal",         "last": "Raleigh"},
    "Carlos Correa":            {"first": "Carlos",      "last": "Correa"},
    "Cedric Mullins":           {"first": "Cedric",      "last": "Mullins"},
    "Chas McCormick":           {"first": "Chas",        "last": "McCormick"},
    "Chris Taylor":             {"first": "Chris",       "last": "Taylor"},
    "Christian Walker":         {"first": "Christian",   "last": "Walker"},
    "Christian Yelich":         {"first": "Christian",   "last": "Yelich"},
    "CJ Abrams":                {"first": "CJ",          "last": "Abrams"},
    "CJ Cron":                  {"first": "CJ",          "last": "Cron",      "mlbam_id": 543068},
    "Cody Bellinger":           {"first": "Cody",        "last": "Bellinger"},
    "Corey Seager":             {"first": "Corey",       "last": "Seager"},
    "Dansby Swanson":           {"first": "Dansby",      "last": "Swanson"},
    "Daulton Varsho":           {"first": "Daulton",     "last": "Varsho"},
    "DJ LeMahieu":              {"first": "DJ",          "last": "LeMahieu"},
    "Dylan Carlson":            {"first": "Dylan",       "last": "Carlson"},
    "Eddie Rosario":            {"first": "Eddie",       "last": "Rosario"},
    "Eduardo Escobar":          {"first": "Eduardo",     "last": "Escobar"},
    "Eloy Jimenez":             {"first": "Eloy",        "last": "Jimenez",   "mlbam_id": 650391},
    "Elly De La Cruz":          {"first": "Elly",        "last": "De La Cruz"},
    "Eugenio Suarez":           {"first": "Eugenio",     "last": "Suarez",    "mlbam_id": 553993},
    "Fernando Tatis Jr.":       {"first": "Fernando",    "last": "Tatis",     "mlbam_id": 665487},
    "Francisco Lindor":         {"first": "Francisco",   "last": "Lindor"},
    "Freddie Freeman":          {"first": "Freddie",     "last": "Freeman"},
    "Gavin Lux":                {"first": "Gavin",       "last": "Lux"},
    "George Springer":          {"first": "George",      "last": "Springer"},
    "Geraldo Perdomo":          {"first": "Geraldo",     "last": "Perdomo"},
    "Giancarlo Stanton":        {"first": "Giancarlo",   "last": "Stanton"},
    "Gleyber Torres":           {"first": "Gleyber",     "last": "Torres"},
    "Gunnar Henderson":         {"first": "Gunnar",      "last": "Henderson"},
    "Ha-Seong Kim":             {"first": "Ha-Seong",    "last": "Kim"},
    "Harrison Bader":           {"first": "Harrison",    "last": "Bader"},
    "Hunter Renfroe":           {"first": "Hunter",      "last": "Renfroe"},
    "Ian Happ":                 {"first": "Ian",         "last": "Happ"},
    "Isaac Paredes":            {"first": "Isaac",       "last": "Paredes"},
    "Isiah Kiner-Falefa":       {"first": "Isiah",       "last": "Kiner-Falefa"},
    "J.D. Martinez":            {"first": "J.D.",        "last": "Martinez",  "mlbam_id": 502110},
    "Jackson Chourio":          {"first": "Jackson",     "last": "Chourio"},
    "Jackson Merrill":          {"first": "Jackson",     "last": "Merrill"},
    "Jake Cronenworth":         {"first": "Jake",        "last": "Cronenworth"},
    "Jake Fraley":              {"first": "Jake",        "last": "Fraley"},
    "Jarred Kelenic":           {"first": "Jarred",      "last": "Kelenic"},
    "Jarren Duran":             {"first": "Jarren",      "last": "Duran"},
    "Javier Baez":              {"first": "Javier",      "last": "Baez",      "mlbam_id": 595879},
    "Jazz Chisholm Jr.":        {"first": "Jazz",        "last": "Chisholm"},
    "Jeff McNeil":              {"first": "Jeff",        "last": "McNeil"},
    "Jeimer Candelario":        {"first": "Jeimer",      "last": "Candelario"},
    "Jeremy Pena":              {"first": "Jeremy",      "last": "Pena",      "mlbam_id": 665161},
    "Jesse Winker":             {"first": "Jesse",       "last": "Winker"},
    "Joc Pederson":             {"first": "Joc",         "last": "Pederson"},
    "Joey Gallo":               {"first": "Joey",        "last": "Gallo"},
    "Jonah Heim":               {"first": "Jonah",       "last": "Heim"},
    "Jonathan India":           {"first": "Jonathan",    "last": "India"},
    "Jorge Soler":              {"first": "Jorge",       "last": "Soler"},
    "Jose Abreu":               {"first": "Jose",        "last": "Abreu",     "mlbam_id": 547989},
    "Jose Altuve":              {"first": "Jose",        "last": "Altuve",    "mlbam_id": 514888},
    "Jose Ramirez":             {"first": "Jose",        "last": "Ramirez",   "mlbam_id": 608070},
    "JP Crawford":              {"first": "J.P.",        "last": "Crawford",  "mlbam_id": 641487},
    "Juan Soto":                {"first": "Juan",        "last": "Soto"},
    "Julio Rodriguez":          {"first": "Julio",       "last": "Rodriguez", "mlbam_id": 677594},
    "Jurickson Profar":         {"first": "Jurickson",   "last": "Profar"},
    "Ke'Bryan Hayes":           {"first": "Ke'Bryan",    "last": "Hayes"},
    "Ketel Marte":              {"first": "Ketel",       "last": "Marte"},
    "Kolten Wong":              {"first": "Kolten",      "last": "Wong"},
    "Kris Bryant":              {"first": "Kris",        "last": "Bryant"},
    "Kyle Schwarber":           {"first": "Kyle",        "last": "Schwarber"},
    "Kyle Tucker":              {"first": "Kyle",        "last": "Tucker"},
    "Lane Thomas":              {"first": "Lane",        "last": "Thomas"},
    "Lars Nootbaar":            {"first": "Lars",        "last": "Nootbaar"},
    "Lourdes Gurriel Jr.":      {"first": "Lourdes",     "last": "Gurriel"},
    "Luis Arraez":              {"first": "Luis",        "last": "Arraez",    "mlbam_id": 650333},
    "Luis Rengifo":             {"first": "Luis",        "last": "Rengifo"},
    "Luis Robert":              {"first": "Luis",        "last": "Robert"},
    "Luke Voit":                {"first": "Luke",        "last": "Voit"},
    "Manny Machado":            {"first": "Manny",       "last": "Machado"},
    "Marcell Ozuna":            {"first": "Marcell",     "last": "Ozuna"},
    "Marcus Semien":            {"first": "Marcus",      "last": "Semien"},
    "Masataka Yoshida":         {"first": "Masataka",    "last": "Yoshida"},
    "Matt Olson":               {"first": "Matt",        "last": "Olson"},
    "Max Muncy":                {"first": "Max",         "last": "Muncy"},
    "Michael Brantley":         {"first": "Michael",     "last": "Brantley"},
    "Michael Conforto":         {"first": "Michael",     "last": "Conforto"},
    "Michael Harris II":        {"first": "Michael",     "last": "Harris"},
    "Mike Trout":               {"first": "Mike",        "last": "Trout"},
    "MJ Melendez":              {"first": "MJ",          "last": "Melendez"},
    "Mookie Betts":             {"first": "Mookie",      "last": "Betts"},
    "Nathaniel Lowe":           {"first": "Nathaniel",   "last": "Lowe"},
    "Nick Castellanos":         {"first": "Nick",        "last": "Castellanos"},
    "Nico Hoerner":             {"first": "Nico",        "last": "Hoerner"},
    "Nolan Arenado":            {"first": "Nolan",       "last": "Arenado"},
    "Nolan Gorman":             {"first": "Nolan",       "last": "Gorman"},
    "Oneil Cruz":               {"first": "Oneil",       "last": "Cruz"},
    "Ozzie Albies":             {"first": "Ozzie",       "last": "Albies"},
    "Paul Goldschmidt":         {"first": "Paul",        "last": "Goldschmidt"},
    "Pete Alonso":              {"first": "Pete",        "last": "Alonso"},
    "Rafael Devers":            {"first": "Rafael",      "last": "Devers"},
    "Ramon Laureano":           {"first": "Ramon",       "last": "Laureano",  "mlbam_id": 657656},
    "Randy Arozarena":          {"first": "Randy",       "last": "Arozarena"},
    "Ronald Acuna Jr.":         {"first": "Ronald",      "last": "Acuna",     "mlbam_id": 660670},
    "Rowdy Tellez":             {"first": "Rowdy",       "last": "Tellez"},
    "Ryan McMahon":             {"first": "Ryan",        "last": "McMahon"},
    "Ryan Mountcastle":         {"first": "Ryan",        "last": "Mountcastle"},
    "Salvador Perez":           {"first": "Salvador",    "last": "Perez",     "mlbam_id": 521692},
    "Sean Murphy":              {"first": "Sean",        "last": "Murphy"},
    "Seiya Suzuki":             {"first": "Seiya",       "last": "Suzuki"},
    "Shohei Ohtani":            {"first": "Shohei",      "last": "Ohtani"},
    "Spencer Steer":            {"first": "Spencer",     "last": "Steer"},
    "Spencer Torkelson":        {"first": "Spencer",     "last": "Torkelson"},
    "Starling Marte":           {"first": "Starling",    "last": "Marte"},
    "Steven Kwan":              {"first": "Steven",      "last": "Kwan"},
    "Taylor Ward":              {"first": "Taylor",      "last": "Ward"},
    "Teoscar Hernandez":        {"first": "Teoscar",     "last": "Hernandez", "mlbam_id": 606192},
    "Tim Anderson":             {"first": "Tim",         "last": "Anderson"},
    "TJ Friedl":                {"first": "TJ",          "last": "Friedl"},
    "Tommy Edman":              {"first": "Tommy",       "last": "Edman"},
    "Tommy Pham":               {"first": "Tommy",       "last": "Pham"},
    "Travis d'Arnaud":          {"first": "Travis",      "last": "d'Arnaud"},
    "Trea Turner":              {"first": "Trea",        "last": "Turner"},
    "Trent Grisham":            {"first": "Trent",       "last": "Grisham"},
    "Trey Mancini":             {"first": "Trey",        "last": "Mancini"},
    "Ty France":                {"first": "Ty",          "last": "France"},
    "Tyler O'Neill":            {"first": "Tyler",       "last": "O'Neill"},
    "Tyler Stephenson":         {"first": "Tyler",       "last": "Stephenson"},
    "Victor Robles":            {"first": "Victor",      "last": "Robles",    "mlbam_id": 645302},
    "Vladimir Guerrero Jr.":    {"first": "Vladimir",    "last": "Guerrero"},
    "Whit Merrifield":          {"first": "Whit",        "last": "Merrifield"},
    "Will Smith":               {"first": "Will",        "last": "Smith",     "mlbam_id": 669257},
    "Willson Contreras":        {"first": "Willson",     "last": "Contreras"},
    "Willy Adames":             {"first": "Willy",       "last": "Adames"},
    "Wilmer Flores":            {"first": "Wilmer",      "last": "Flores"},
    "Xander Bogaerts":          {"first": "Xander",      "last": "Bogaerts"},
    "Yandy Diaz":               {"first": "Yandy",       "last": "Diaz",      "mlbam_id": 650490},
    "Yordan Alvarez":           {"first": "Yordan",      "last": "Alvarez",   "mlbam_id": 670541},
}

SWING_EVENTS = {
    "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
}

# ── Classification (mirrors dashboard.py) ─────────────────────────────────────
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


def pull_with_retries(season, mlbam_id, label):
    """Pulls one player-season, retrying up to MAX_RETRIES times on empty/error."""
    last_raw = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = statcast_batter(f"{season}-03-01", f"{season}-11-30", mlbam_id)
            if raw is not None and not raw.empty:
                return raw
            print(f"  [empty attempt {attempt}]  {label}")
        except Exception as e:
            print(f"  [error attempt {attempt}]  {label}: {e}")
        time.sleep(2 ** attempt)  # exponential backoff: 2s, 4s, 8s
    return last_raw


# ── Pull loop ──────────────────────────────────────────────────────────────────
OUT_PATH = "C:/Users/jaico/baseball/barrel_data.parquet"

total   = len(PLAYERS) * len(SEASONS)
done    = 0
chunks  = []
skipped = []

print(f"Pulling {len(PLAYERS)} players × {len(SEASONS)} seasons = {total} requests\n")

for player_name, info in PLAYERS.items():
    # Use hardcoded mlbam_id if provided, otherwise resolve via lookup
    if "mlbam_id" in info:
        mlbam_id = info["mlbam_id"]
    else:
        try:
            lkp = playerid_lookup(info["last"], info["first"])
            if lkp.empty:
                print(f"  [no ID]  {player_name}")
                skipped.append(player_name)
                done += len(SEASONS)
                continue
            mlbam_id = int(lkp.iloc[0]["key_mlbam"])
        except Exception as e:
            print(f"  [lookup error] {player_name}: {e}")
            skipped.append(player_name)
            done += len(SEASONS)
            continue

    for season in SEASONS:
        done += 1
        pct   = done / total * 100
        label = f"[{done:>3}/{total}] {pct:5.1f}%  {player_name} {season}"
        try:
            raw = pull_with_retries(season, mlbam_id, label)
            if raw is None or (hasattr(raw, 'empty') and raw.empty):
                print(f"  [empty]  {label}")
                continue
            df = classify(raw, player_name, season)
            chunks.append(df)
            print(f"  [ok]     {label}  — {len(df):,} pitches")
        except Exception as e:
            print(f"  [error]  {label}: {e}")

# ── Combine & save ─────────────────────────────────────────────────────────────
if not chunks:
    print("\nNo data collected — nothing saved.")
else:
    print(f"\nCombining {len(chunks)} chunks …")
    combined = pd.concat(chunks, ignore_index=True)

    # Tidy up dtypes to keep parquet compact
    for col in ["is_swing", "is_barrel"]:
        combined[col] = combined[col].astype(bool)
    combined["season"] = combined["season"].astype("int16")

    print(f"Total rows : {len(combined):,}")
    print(f"Columns    : {len(combined.columns)}")
    print(f"Saving to  : {OUT_PATH}")
    combined.to_parquet(OUT_PATH, index=False, engine="pyarrow", compression="snappy")
    size_mb = __import__("os").path.getsize(OUT_PATH) / 1_048_576
    print(f"Done — {size_mb:.1f} MB written.")

if skipped:
    print(f"\nSkipped ({len(skipped)}): {', '.join(skipped)}")
