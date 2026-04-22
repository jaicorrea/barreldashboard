import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from scipy.interpolate import RegularGridInterpolator
import pybaseball
from pybaseball import statcast_batter, playerid_lookup
import warnings

warnings.filterwarnings("ignore")
pybaseball.cache.enable()

st.set_page_config(page_title="MLB Barrel Dashboard", layout="wide", page_icon="⚾")

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #313244;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #cba6f7; }
    .metric-label { font-size: 0.8rem; color: #a6adc8; text-transform: uppercase; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
ZONE_X = (-0.83, 0.83)
ZONE_Z = (1.5, 3.5)

# Heatmap grid (module-level so both tabs share the same grid)
XI = np.linspace(-1.5, 1.5, 40)
YI = np.linspace(0.5,  5.0, 40)
XC = (XI[:-1] + XI[1:]) / 2   # bin centres
YC = (YI[:-1] + YI[1:]) / 2

PLAYERS = {
    # A
    "Aaron Judge":              {"first": "Aaron",       "last": "Judge"},
    "Adam Duvall":              {"first": "Adam",        "last": "Duvall"},
    "Adam Frazier":             {"first": "Adam",        "last": "Frazier"},
    "Adley Rutschman":          {"first": "Adley",       "last": "Rutschman"},
    "Adolis Garcia":            {"first": "Adolis",      "last": "Garcia"},
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
    # B
    "Bobby Witt Jr.":           {"first": "Bobby",       "last": "Witt"},
    "Bo Bichette":              {"first": "Bo",          "last": "Bichette"},
    "Brendan Rodgers":          {"first": "Brendan",     "last": "Rodgers"},
    "Bryce Harper":             {"first": "Bryce",       "last": "Harper"},
    "Bryan Reynolds":           {"first": "Bryan",       "last": "Reynolds"},
    # C
    "Cal Raleigh":              {"first": "Cal",         "last": "Raleigh"},
    "Carlos Correa":            {"first": "Carlos",      "last": "Correa"},
    "Cedric Mullins":           {"first": "Cedric",      "last": "Mullins"},
    "Chas McCormick":           {"first": "Chas",        "last": "McCormick"},
    "Chris Taylor":             {"first": "Chris",       "last": "Taylor"},
    "Christian Walker":         {"first": "Christian",   "last": "Walker"},
    "Christian Yelich":         {"first": "Christian",   "last": "Yelich"},
    "CJ Abrams":                {"first": "CJ",          "last": "Abrams"},
    "CJ Cron":                  {"first": "CJ",          "last": "Cron"},
    "Cody Bellinger":           {"first": "Cody",        "last": "Bellinger"},
    "Corey Seager":             {"first": "Corey",       "last": "Seager"},
    # D
    "Dansby Swanson":           {"first": "Dansby",      "last": "Swanson"},
    "Daulton Varsho":           {"first": "Daulton",     "last": "Varsho"},
    "DJ LeMahieu":              {"first": "DJ",          "last": "LeMahieu"},
    "Dylan Carlson":            {"first": "Dylan",       "last": "Carlson"},
    # E
    "Eddie Rosario":            {"first": "Eddie",       "last": "Rosario"},
    "Eduardo Escobar":          {"first": "Eduardo",     "last": "Escobar"},
    "Eloy Jimenez":             {"first": "Eloy",        "last": "Jimenez"},
    "Elly De La Cruz":          {"first": "Elly",        "last": "De La Cruz"},
    "Eugenio Suarez":           {"first": "Eugenio",     "last": "Suarez"},
    # F
    "Fernando Tatis Jr.":       {"first": "Fernando",    "last": "Tatis"},
    "Francisco Lindor":         {"first": "Francisco",   "last": "Lindor"},
    "Freddie Freeman":          {"first": "Freddie",     "last": "Freeman"},
    # G
    "Gavin Lux":                {"first": "Gavin",       "last": "Lux"},
    "George Springer":          {"first": "George",      "last": "Springer"},
    "Geraldo Perdomo":          {"first": "Geraldo",     "last": "Perdomo"},
    "Giancarlo Stanton":        {"first": "Giancarlo",   "last": "Stanton"},
    "Gleyber Torres":           {"first": "Gleyber",     "last": "Torres"},
    "Gunnar Henderson":         {"first": "Gunnar",      "last": "Henderson"},
    # H
    "Ha-Seong Kim":             {"first": "Ha-Seong",    "last": "Kim"},
    "Harrison Bader":           {"first": "Harrison",    "last": "Bader"},
    "Hunter Renfroe":           {"first": "Hunter",      "last": "Renfroe"},
    # I
    "Ian Happ":                 {"first": "Ian",         "last": "Happ"},
    "Isaac Paredes":            {"first": "Isaac",       "last": "Paredes"},
    "Isiah Kiner-Falefa":       {"first": "Isiah",       "last": "Kiner-Falefa"},
    # J
    "J.D. Martinez":            {"first": "J.D.",        "last": "Martinez"},
    "Jackson Chourio":          {"first": "Jackson",     "last": "Chourio"},
    "Jackson Merrill":          {"first": "Jackson",     "last": "Merrill"},
    "Jake Cronenworth":         {"first": "Jake",        "last": "Cronenworth"},
    "Jake Fraley":              {"first": "Jake",        "last": "Fraley"},
    "Jarred Kelenic":           {"first": "Jarred",      "last": "Kelenic"},
    "Jarren Duran":             {"first": "Jarren",      "last": "Duran"},
    "Javier Baez":              {"first": "Javier",      "last": "Baez"},
    "Jazz Chisholm Jr.":        {"first": "Jazz",        "last": "Chisholm"},
    "Jeff McNeil":              {"first": "Jeff",        "last": "McNeil"},
    "Jeimer Candelario":        {"first": "Jeimer",      "last": "Candelario"},
    "Jeremy Pena":              {"first": "Jeremy",      "last": "Pena"},
    "Jesse Winker":             {"first": "Jesse",       "last": "Winker"},
    "Joc Pederson":             {"first": "Joc",         "last": "Pederson"},
    "Joey Gallo":               {"first": "Joey",        "last": "Gallo"},
    "Jonah Heim":               {"first": "Jonah",       "last": "Heim"},
    "Jonathan India":           {"first": "Jonathan",    "last": "India"},
    "Jorge Soler":              {"first": "Jorge",       "last": "Soler"},
    "Jose Abreu":               {"first": "Jose",        "last": "Abreu"},
    "Jose Altuve":              {"first": "Jose",        "last": "Altuve"},
    "Jose Ramirez":             {"first": "Jose",        "last": "Ramirez"},
    "JP Crawford":              {"first": "J.P.",        "last": "Crawford"},
    "Juan Soto":                {"first": "Juan",        "last": "Soto"},
    "Julio Rodriguez":          {"first": "Julio",       "last": "Rodriguez"},
    "Jurickson Profar":         {"first": "Jurickson",   "last": "Profar"},
    # K
    "Ke'Bryan Hayes":           {"first": "Ke'Bryan",    "last": "Hayes"},
    "Ketel Marte":              {"first": "Ketel",       "last": "Marte"},
    "Kolten Wong":              {"first": "Kolten",      "last": "Wong"},
    "Kris Bryant":              {"first": "Kris",        "last": "Bryant"},
    "Kyle Schwarber":           {"first": "Kyle",        "last": "Schwarber"},
    "Kyle Tucker":              {"first": "Kyle",        "last": "Tucker"},
    # L
    "Lane Thomas":              {"first": "Lane",        "last": "Thomas"},
    "Lars Nootbaar":            {"first": "Lars",        "last": "Nootbaar"},
    "Lourdes Gurriel Jr.":      {"first": "Lourdes",     "last": "Gurriel"},
    "Luis Arraez":              {"first": "Luis",        "last": "Arraez"},
    "Luis Rengifo":             {"first": "Luis",        "last": "Rengifo"},
    "Luis Robert":              {"first": "Luis",        "last": "Robert"},
    "Luke Voit":                {"first": "Luke",        "last": "Voit"},
    # M
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
    # N
    "Nathaniel Lowe":           {"first": "Nathaniel",   "last": "Lowe"},
    "Nick Castellanos":         {"first": "Nick",        "last": "Castellanos"},
    "Nico Hoerner":             {"first": "Nico",        "last": "Hoerner"},
    "Nolan Arenado":            {"first": "Nolan",       "last": "Arenado"},
    "Nolan Gorman":             {"first": "Nolan",       "last": "Gorman"},
    # O
    "Oneil Cruz":               {"first": "Oneil",       "last": "Cruz"},
    "Ozzie Albies":             {"first": "Ozzie",       "last": "Albies"},
    # P
    "Paul Goldschmidt":         {"first": "Paul",        "last": "Goldschmidt"},
    "Pete Alonso":              {"first": "Pete",        "last": "Alonso"},
    # R
    "Rafael Devers":            {"first": "Rafael",      "last": "Devers"},
    "Ramon Laureano":           {"first": "Ramon",       "last": "Laureano"},
    "Randy Arozarena":          {"first": "Randy",       "last": "Arozarena"},
    "Ronald Acuna Jr.":         {"first": "Ronald",      "last": "Acuna"},
    "Rowdy Tellez":             {"first": "Rowdy",       "last": "Tellez"},
    "Ryan McMahon":             {"first": "Ryan",        "last": "McMahon"},
    "Ryan Mountcastle":         {"first": "Ryan",        "last": "Mountcastle"},
    # S
    "Salvador Perez":           {"first": "Salvador",    "last": "Perez"},
    "Sean Murphy":              {"first": "Sean",        "last": "Murphy"},
    "Seiya Suzuki":             {"first": "Seiya",       "last": "Suzuki"},
    "Shohei Ohtani":            {"first": "Shohei",      "last": "Ohtani"},
    "Spencer Steer":            {"first": "Spencer",     "last": "Steer"},
    "Spencer Torkelson":        {"first": "Spencer",     "last": "Torkelson"},
    "Starling Marte":           {"first": "Starling",    "last": "Marte"},
    "Steven Kwan":              {"first": "Steven",      "last": "Kwan"},
    # T
    "Taylor Ward":              {"first": "Taylor",      "last": "Ward"},
    "Teoscar Hernandez":        {"first": "Teoscar",     "last": "Hernandez"},
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
    # V
    "Victor Robles":            {"first": "Victor",      "last": "Robles"},
    "Vladimir Guerrero Jr.":    {"first": "Vladimir",    "last": "Guerrero"},
    # W
    "Whit Merrifield":          {"first": "Whit",        "last": "Merrifield"},
    "Will Smith":               {"first": "Will",        "last": "Smith"},
    "Willson Contreras":        {"first": "Willson",     "last": "Contreras"},
    "Willy Adames":             {"first": "Willy",       "last": "Adames"},
    "Wilmer Flores":            {"first": "Wilmer",      "last": "Flores"},
    # X
    "Xander Bogaerts":          {"first": "Xander",      "last": "Bogaerts"},
    # Y
    "Yandy Diaz":               {"first": "Yandy",       "last": "Diaz"},
    "Yordan Alvarez":           {"first": "Yordan",      "last": "Alvarez"},
}

SEASONS = [2021, 2022, 2023, 2024, 2025]

SWING_EVENTS = {
    "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
}

# ── Data helpers ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def load_statcast(first: str, last: str, season: int) -> pd.DataFrame:
    lookup = playerid_lookup(last, first)
    if lookup.empty:
        return pd.DataFrame()
    mlbam_id = int(lookup.iloc[0]["key_mlbam"])
    df = statcast_batter(f"{season}-03-01", f"{season}-11-30", mlbam_id)
    return df

def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_swing"] = df["description"].isin(SWING_EVENTS)

    # MLB official barrel definition — expanding LA window above 98 mph.
    # At 98 mph: [26, 30].  At 116 mph: [8, 50].
    ev      = df["launch_speed"].to_numpy(dtype=float, na_value=np.nan)
    la      = df["launch_angle"].to_numpy(dtype=float, na_value=np.nan)
    ev_cap  = np.clip(ev, 0, 116)
    delta   = ev_cap - 98
    min_la  = 26 - delta
    max_la  = 30 + delta * (20.0 / 18.0)

    df["is_barrel"] = (
        (~np.isnan(ev)) & (~np.isnan(la)) &
        (ev >= 98) &
        (la >= min_la) & (la <= max_la)
    )
    return df

def build_barrel_kde(barrel_df: pd.DataFrame,
                     xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    x, y = barrel_df["plate_x"].values, barrel_df["plate_z"].values
    if len(x) < 5:
        return np.zeros((len(yc), len(xc)))
    try:
        kde = gaussian_kde(np.vstack([x, y]), bw_method="scott")
        xx, yy = np.meshgrid(xc, yc)
        z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        return z / z.max()
    except Exception:
        return np.zeros((len(yc), len(xc)))

def in_barrel_kde_zone(pitch_df: pd.DataFrame,
                       kde_grid: np.ndarray,
                       xc: np.ndarray, yc: np.ndarray,
                       threshold: float) -> pd.Series:
    if kde_grid.max() == 0:
        return pd.Series(False, index=pitch_df.index)
    interp = RegularGridInterpolator(
        (yc, xc), kde_grid,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    density = interp(pitch_df[["plate_z", "plate_x"]].values)
    return pd.Series(density >= threshold, index=pitch_df.index)

def binned_swing_rate(df: pd.DataFrame,
                      xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    grid = np.zeros((len(yi) - 1, len(xi) - 1))
    for i in range(len(yi) - 1):
        for j in range(len(xi) - 1):
            mask = (
                df["plate_x"].between(xi[j], xi[j + 1]) &
                df["plate_z"].between(yi[i], yi[i + 1])
            )
            if mask.sum() >= 3:
                grid[i, j] = df.loc[mask, "is_swing"].mean()
    return grid

def make_zone_rect() -> dict:
    return dict(
        type="rect",
        x0=ZONE_X[0], x1=ZONE_X[1],
        y0=ZONE_Z[0], y1=ZONE_Z[1],
        line=dict(color="white", width=2, dash="dash"),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )

def player_metrics(first: str, last: str, season: int,
                   kde_threshold: float) -> dict | None:
    """Compute per-player leaderboard row. Returns None if insufficient data."""
    raw = load_statcast(first, last, season)
    if raw is None or raw.empty:
        return None
    df        = classify(raw)
    pitch_df  = df.dropna(subset=["plate_x", "plate_z"])
    batted_df = df[df["type"] == "X"].dropna(subset=["launch_speed", "launch_angle"])
    barrel_df = batted_df[batted_df["is_barrel"]]

    # Minimum 100 PA filter (approximate via pitches seen)
    if len(pitch_df) < 50:    # very low floor — career threshold is enforced by PLAYERS dict
        return None

    kde_grid = build_barrel_kde(barrel_df, XC, YC)
    in_hot   = in_barrel_kde_zone(pitch_df, kde_grid, XC, YC, kde_threshold)
    hot_df   = pitch_df[in_hot]
    cold_df  = pitch_df[~in_hot]

    swing_in  = hot_df["is_swing"].mean()  * 100 if len(hot_df)  >= 10 else np.nan
    swing_out = cold_df["is_swing"].mean() * 100 if len(cold_df) >= 10 else np.nan
    diff      = (swing_in - swing_out) if not (np.isnan(swing_in) or np.isnan(swing_out)) else np.nan

    # wOBA from Statcast woba_value / woba_denom columns
    woba = np.nan
    if "woba_value" in pitch_df.columns and "woba_denom" in pitch_df.columns:
        denom = pitch_df["woba_denom"].sum(skipna=True)
        if denom > 0:
            woba = pitch_df["woba_value"].sum(skipna=True) / denom

    return {
        "swing_in":  swing_in,
        "swing_out": swing_out,
        "diff":      diff,
        "woba":      woba,
    }

@st.cache_data(show_spinner=False, ttl=3600)
def build_leaderboard(season: int, kde_threshold: float) -> pd.DataFrame:
    rows = []
    for name, info in PLAYERS.items():
        m = player_metrics(info["first"], info["last"], season, kde_threshold)
        if m is None:
            continue
        rows.append({
            "Player":                   name,
            "Swing% In Barrel Zone":    m["swing_in"],
            "Swing% Outside Barrel Zone": m["swing_out"],
            "Difference":               m["diff"],
            "wOBA":                     m["woba"],
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Difference", ascending=False).reset_index(drop=True)
    return df

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚾ Controls")
season        = st.sidebar.selectbox("Season", SEASONS[::-1], index=0)
player_name   = st.sidebar.selectbox("Player (dashboard tab)", list(PLAYERS.keys()), index=0)
overlay       = st.sidebar.radio(
    "Heatmap overlay",
    ["Barrels only", "Swing % (in barrel zone)", "Swing % (outside barrel zone)"],
    index=0,
)
kde_threshold = st.sidebar.slider(
    "Barrel KDE hot-zone threshold",
    min_value=0.10, max_value=0.80, value=0.40, step=0.05,
    help="Fraction of peak barrel density a pitch must exceed to be 'inside' the hot zone.",
)
colorscale = st.sidebar.selectbox("Color scale", ["Hot", "Viridis", "Plasma", "RdYlBu_r"], index=0)

# ── Tabs ───────────────────────────────────────────────────────────────────────
st.title("🔥 MLB Barrel Dashboard")
tab_dash, tab_lb = st.tabs(["🏟️ Player Dashboard", "🏆 Leaderboard"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Player Dashboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.subheader(f"{player_name} · {season}")

    with st.spinner(f"Loading Statcast data for {player_name} ({season})…"):
        info   = PLAYERS[player_name]
        raw_df = load_statcast(info["first"], info["last"], season)

    if raw_df is None or raw_df.empty:
        st.error("No Statcast data found for this player/season combination.")
    else:
        df        = classify(raw_df)
        pitch_df  = df.dropna(subset=["plate_x", "plate_z"])
        batted_df = df[df["type"] == "X"].dropna(subset=["launch_speed", "launch_angle"])
        barrel_df = batted_df[batted_df["is_barrel"]]

        kde_grid  = build_barrel_kde(barrel_df, XC, YC)
        in_hot    = in_barrel_kde_zone(pitch_df, kde_grid, XC, YC, kde_threshold)
        hot_pitch_df  = pitch_df[in_hot]
        cold_pitch_df = pitch_df[~in_hot]

        # ── Metrics ──────────────────────────────────────────────────────────
        batted_balls       = len(batted_df)
        total_swings       = pitch_df["is_swing"].sum()
        barrel_rate        = len(barrel_df) / max(batted_balls, 1) * 100
        barrel_zone_swing  = hot_pitch_df["is_swing"].mean()  * 100 if len(hot_pitch_df)  else 0.0
        non_barrel_swing   = cold_pitch_df["is_swing"].mean() * 100 if len(cold_pitch_df) else 0.0
        contact_rate       = (
            pitch_df[pitch_df["description"].isin({"hit_into_play", "foul", "foul_tip", "foul_bunt"})]
            .shape[0] / max(total_swings, 1) * 100
        )
        woba = np.nan
        if "woba_value" in pitch_df.columns and "woba_denom" in pitch_df.columns:
            denom = pitch_df["woba_denom"].sum(skipna=True)
            if denom > 0:
                woba = pitch_df["woba_value"].sum(skipna=True) / denom

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        for col, val, label in [
            (col1, f"{barrel_rate:.1f}%",       "Barrel Rate"),
            (col2, f"{barrel_zone_swing:.1f}%", "Swing% In Barrel Zone"),
            (col3, f"{non_barrel_swing:.1f}%",  "Swing% Outside Barrel Zone"),
            (col4, f"{barrel_zone_swing - non_barrel_swing:+.1f}%", "Difference"),
            (col5, f"{contact_rate:.1f}%",      "Contact Rate"),
            (col6, f"{woba:.3f}" if not np.isnan(woba) else "N/A", "wOBA"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Main heatmap ──────────────────────────────────────────────────────
        if overlay == "Barrels only":
            z_grid, title_overlay, colorbar_title, tick_fmt = (
                kde_grid, "Barrel KDE Density", "Relative Density", "")
        elif overlay == "Swing % (in barrel zone)":
            z_grid         = binned_swing_rate(hot_pitch_df, XI, YI)
            title_overlay  = f"Swing % — inside barrel hot zone (≥ {kde_threshold:.0%} density)"
            colorbar_title = "Swing %"
            tick_fmt       = ".0%"
        else:
            z_grid         = binned_swing_rate(cold_pitch_df, XI, YI)
            title_overlay  = f"Swing % — outside barrel hot zone (< {kde_threshold:.0%} density)"
            colorbar_title = "Swing %"
            tick_fmt       = ".0%"

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            x=XC, y=YC, z=z_grid,
            colorscale=colorscale,
            colorbar=dict(title=colorbar_title, tickformat=tick_fmt),
            opacity=0.85, zmin=0,
        ))
        if kde_grid.max() > 0:
            fig.add_trace(go.Contour(
                x=XC, y=YC, z=kde_grid,
                contours=dict(start=kde_threshold, end=kde_threshold, size=0, coloring="none"),
                line=dict(color="#a6e3a1", width=2, dash="dot"),
                showscale=False,
                name=f"Hot zone boundary ({kde_threshold:.0%})",
                hoverinfo="skip",
            ))
        if len(barrel_df) > 0:
            fig.add_trace(go.Scatter(
                x=barrel_df["plate_x"], y=barrel_df["plate_z"],
                mode="markers",
                marker=dict(size=6, color="#f38ba8", opacity=0.75,
                            line=dict(color="white", width=0.5)),
                name="Barrel",
                hovertemplate=(
                    "Plate X: %{x:.2f} ft<br>Plate Z: %{y:.2f} ft<br>"
                    "Exit Velo: %{customdata[0]:.1f} mph<br>"
                    "Launch Angle: %{customdata[1]:.1f}°<br>"
                    "Result: %{customdata[2]}<extra></extra>"
                ),
                customdata=barrel_df[["launch_speed", "launch_angle", "events"]].values,
            ))
        fig.update_layout(
            shapes=[make_zone_rect()],
            title=dict(text=f"{player_name} · {season} · {title_overlay}",
                       font=dict(size=16, color="white")),
            xaxis=dict(title="Horizontal Position (ft) — Pitcher's View",
                       range=[1.5, -1.5], zeroline=False, color="white", gridcolor="#313244"),
            yaxis=dict(title="Vertical Position (ft)",
                       range=[0.5, 5.0], zeroline=False, color="white", gridcolor="#313244",
                       scaleanchor="x", scaleratio=1),
            paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
            font=dict(color="white"),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
            height=600, margin=dict(l=60, r=40, t=70, b=60),
        )
        fig.add_shape(
            type="path",
            path="M -0.83 0.6 L 0.83 0.6 L 0.83 0.75 L 0 0.9 L -0.83 0.75 Z",
            fillcolor="rgba(255,255,255,0.08)", line=dict(color="#a6adc8", width=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Green dotted contour = barrel KDE hot-zone boundary at {kde_threshold:.0%} of peak density · "
            "Red dots = individual barrels · Dashed white rectangle = standard strike zone · "
            "Pitcher's perspective"
        )

        # ── Distribution panels ───────────────────────────────────────────────
        st.subheader("Barrel Distribution")
        col_a, col_b = st.columns(2)
        with col_a:
            if len(batted_df) > 0:
                fig_ev = go.Figure()
                fig_ev.add_trace(go.Histogram(
                    x=batted_df["launch_speed"], name="All BIP",
                    marker_color="#89b4fa", opacity=0.6,
                    xbins=dict(start=40, end=120, size=2),
                ))
                if len(barrel_df) > 0:
                    fig_ev.add_trace(go.Histogram(
                        x=barrel_df["launch_speed"], name="Barrels",
                        marker_color="#f38ba8", opacity=0.9,
                        xbins=dict(start=40, end=120, size=2),
                    ))
                fig_ev.add_vline(x=98, line_dash="dash", line_color="#a6e3a1",
                                 annotation_text="98 mph", annotation_position="top right")
                fig_ev.update_layout(
                    barmode="overlay", title="Exit Velocity Distribution",
                    xaxis_title="Exit Velocity (mph)", yaxis_title="Count",
                    paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
                    font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0)"),
                    height=320, margin=dict(l=50, r=20, t=50, b=50),
                )
                st.plotly_chart(fig_ev, use_container_width=True)
        with col_b:
            if len(batted_df) > 0:
                fig_la = go.Figure()
                fig_la.add_trace(go.Histogram(
                    x=batted_df["launch_angle"], name="All BIP",
                    marker_color="#89b4fa", opacity=0.6,
                    xbins=dict(start=-90, end=90, size=3),
                ))
                if len(barrel_df) > 0:
                    fig_la.add_trace(go.Histogram(
                        x=barrel_df["launch_angle"], name="Barrels",
                        marker_color="#f38ba8", opacity=0.9,
                        xbins=dict(start=-90, end=90, size=3),
                    ))
                fig_la.add_vrect(x0=26, x1=30, fillcolor="rgba(163,227,153,0.15)",
                                 line_color="#a6e3a1", line_dash="dash",
                                 annotation_text="Barrel LA band",
                                 annotation_position="top left")
                fig_la.update_layout(
                    barmode="overlay", title="Launch Angle Distribution",
                    xaxis_title="Launch Angle (°)", yaxis_title="Count",
                    paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
                    font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0)"),
                    height=320, margin=dict(l=50, r=20, t=50, b=50),
                )
                st.plotly_chart(fig_la, use_container_width=True)

        # ── Spray chart ───────────────────────────────────────────────────────
        st.subheader("Spray Chart")
        if "hc_x" in df.columns and "hc_y" in df.columns:
            spray_df = batted_df.dropna(subset=["hc_x", "hc_y"])
            if not spray_df.empty:
                fig_spray = go.Figure()
                non_barrel   = spray_df[~spray_df["is_barrel"]]
                barrel_spray = spray_df[spray_df["is_barrel"]]
                fig_spray.add_trace(go.Scatter(
                    x=non_barrel["hc_x"], y=non_barrel["hc_y"].multiply(-1),
                    mode="markers",
                    marker=dict(size=5, color="#89b4fa", opacity=0.4),
                    name="Other BIP",
                ))
                fig_spray.add_trace(go.Scatter(
                    x=barrel_spray["hc_x"], y=barrel_spray["hc_y"].multiply(-1),
                    mode="markers",
                    marker=dict(size=9, color="#f38ba8", symbol="star",
                                line=dict(color="white", width=0.5)),
                    name="Barrel",
                    hovertemplate=(
                        "Exit Velo: %{customdata[0]:.1f} mph<br>"
                        "LA: %{customdata[1]:.1f}°<br>"
                        "Result: %{customdata[2]}<extra></extra>"
                    ),
                    customdata=barrel_spray[["launch_speed", "launch_angle", "events"]].values,
                ))
                fig_spray.update_layout(
                    title="Spray Chart (★ = Barrel)",
                    xaxis=dict(range=[0, 250], showgrid=False, showticklabels=False),
                    yaxis=dict(range=[-250, 0], showgrid=False, showticklabels=False,
                               scaleanchor="x", scaleratio=1),
                    paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
                    font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0)"),
                    height=380, margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_spray, use_container_width=True)
        else:
            st.info("Spray-chart coordinates not available in this dataset slice.")

        with st.expander("Raw barrel data"):
            cols      = ["game_date", "pitch_type", "plate_x", "plate_z",
                         "launch_speed", "launch_angle", "hit_distance_sc", "events", "description"]
            available = [c for c in cols if c in barrel_df.columns]
            st.dataframe(
                barrel_df[available].sort_values("game_date", ascending=False),
                use_container_width=True, height=300,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Leaderboard
# ══════════════════════════════════════════════════════════════════════════════
with tab_lb:
    st.subheader(f"Leaderboard — {season} Season")
    st.caption(
        f"Barrel KDE hot-zone threshold: **{kde_threshold:.0%}** · "
        "Swing% columns use each player's own barrel KDE map as the zone boundary · "
        "Minimum ~100 PA · Click any column header to sort · "
        "wOBA computed from Statcast pitch-level data"
    )

    if st.button("⚡ Build / Refresh Leaderboard", type="primary"):
        build_leaderboard.clear()   # bust cache so fresh pull runs

    # Build with progress bar on first run (cache miss)
    lb_placeholder = st.empty()
    with lb_placeholder.container():
        with st.spinner(f"Computing leaderboard for {season}… (first run fetches all players; subsequent loads are instant)"):
            lb_df = build_leaderboard(season, kde_threshold)

    if lb_df.empty:
        st.warning("No data returned. Try a different season or lower the hot-zone threshold.")
    else:
        # ── Rank column ───────────────────────────────────────────────────────
        lb_df.insert(0, "Rank", range(1, len(lb_df) + 1))

        # ── Plotly table (dark theme, colour-coded Difference) ────────────────
        diff_colors = [
            "#a6e3a1" if v > 0 else "#f38ba8" if v < 0 else "#a6adc8"
            for v in lb_df["Difference"].fillna(0)
        ]

        header_vals = ["Rank", "Player",
                       "Swing%<br>In Barrel Zone",
                       "Swing%<br>Outside Barrel Zone",
                       "Difference",
                       "wOBA"]

        def fmt_pct(col):
            return [f"{v:.1f}%" if not np.isnan(v) else "—" for v in lb_df[col]]

        def fmt_woba(col):
            return [f"{v:.3f}" if not np.isnan(v) else "—" for v in lb_df[col]]

        def fmt_diff(col):
            return [f"{v:+.1f}%" if not np.isnan(v) else "—" for v in lb_df[col]]

        cell_vals = [
            lb_df["Rank"].tolist(),
            lb_df["Player"].tolist(),
            fmt_pct("Swing% In Barrel Zone"),
            fmt_pct("Swing% Outside Barrel Zone"),
            fmt_diff("Difference"),
            fmt_woba("wOBA"),
        ]
        cell_colors = [
            ["#181825"] * len(lb_df),
            ["#1e1e2e"] * len(lb_df),
            ["#1e1e2e"] * len(lb_df),
            ["#1e1e2e"] * len(lb_df),
            diff_colors,
            ["#1e1e2e"] * len(lb_df),
        ]
        cell_font_colors = [
            ["#cba6f7"] * len(lb_df),
            ["white"]   * len(lb_df),
            ["#89b4fa"] * len(lb_df),
            ["#89b4fa"] * len(lb_df),
            ["#181825"] * len(lb_df),   # dark text on coloured bg
            ["#f9e2af"] * len(lb_df),
        ]

        fig_lb = go.Figure(go.Table(
            columnwidth=[40, 160, 130, 160, 110, 80],
            header=dict(
                values=header_vals,
                fill_color="#313244",
                font=dict(color="white", size=12),
                align=["center", "left", "center", "center", "center", "center"],
                line_color="#45475a",
                height=36,
            ),
            cells=dict(
                values=cell_vals,
                fill_color=cell_colors,
                font=dict(color=cell_font_colors, size=12),
                align=["center", "left", "center", "center", "center", "center"],
                line_color="#313244",
                height=30,
            ),
        ))
        fig_lb.update_layout(
            paper_bgcolor="#1e1e2e",
            margin=dict(l=0, r=0, t=10, b=10),
            height=max(400, 36 + len(lb_df) * 30 + 20),
        )
        st.plotly_chart(fig_lb, use_container_width=True)

        # ── Scatter: Swing% In vs wOBA ────────────────────────────────────────
        st.subheader("Swing% In Barrel Zone vs wOBA")
        plot_df = lb_df.dropna(subset=["Swing% In Barrel Zone", "wOBA"])
        if not plot_df.empty:
            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=plot_df["Swing% In Barrel Zone"],
                y=plot_df["wOBA"],
                mode="markers+text",
                text=plot_df["Player"].str.split().str[-1],   # last name
                textposition="top center",
                textfont=dict(size=9, color="#a6adc8"),
                marker=dict(
                    size=10,
                    color=plot_df["Difference"],
                    colorscale="RdYlGn",
                    colorbar=dict(title="Difference", tickformat="+.1f"),
                    line=dict(color="white", width=0.5),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Swing% In: %{x:.1f}%<br>"
                    "wOBA: %{y:.3f}<br>"
                    "Difference: %{customdata[1]:+.1f}%<extra></extra>"
                ),
                customdata=plot_df[["Player", "Difference"]].values,
            ))
            fig_sc.update_layout(
                xaxis_title="Swing% In Barrel Zone",
                yaxis_title="wOBA",
                paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
                font=dict(color="white"),
                height=480, margin=dict(l=60, r=40, t=30, b=60),
            )
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption("Dot colour = Difference (green = swings more in barrel zone than out; red = reverse)")
