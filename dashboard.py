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

# ── Strike zone outline (display only, not used for zone classification) ───────
ZONE_X = (-0.83, 0.83)
ZONE_Z = (1.5, 3.5)

PLAYERS = {
    # A
    "Aaron Judge":              {"first": "Aaron",      "last": "Judge"},
    "Adley Rutschman":          {"first": "Adley",      "last": "Rutschman"},
    "Adolis Garcia":            {"first": "Adolis",     "last": "Garcia"},
    "Alex Bregman":             {"first": "Alex",       "last": "Bregman"},
    "Anthony Rizzo":            {"first": "Anthony",    "last": "Rizzo"},
    "Anthony Santander":        {"first": "Anthony",    "last": "Santander"},
    "Anthony Volpe":            {"first": "Anthony",    "last": "Volpe"},
    "Austin Riley":             {"first": "Austin",     "last": "Riley"},
    # B
    "Bobby Witt Jr.":           {"first": "Bobby",      "last": "Witt"},
    "Bo Bichette":              {"first": "Bo",         "last": "Bichette"},
    "Bryce Harper":             {"first": "Bryce",      "last": "Harper"},
    "Bryan Reynolds":           {"first": "Bryan",      "last": "Reynolds"},
    # C
    "Cal Raleigh":              {"first": "Cal",        "last": "Raleigh"},
    "Carlos Correa":            {"first": "Carlos",     "last": "Correa"},
    "Christian Yelich":         {"first": "Christian",  "last": "Yelich"},
    "Cody Bellinger":           {"first": "Cody",       "last": "Bellinger"},
    "Corey Seager":             {"first": "Corey",      "last": "Seager"},
    "CJ Abrams":                {"first": "CJ",         "last": "Abrams"},
    # D
    "Dansby Swanson":           {"first": "Dansby",     "last": "Swanson"},
    "DJ LeMahieu":              {"first": "DJ",         "last": "LeMahieu"},
    # E
    "Elly De La Cruz":          {"first": "Elly",       "last": "De La Cruz"},
    "Eugenio Suarez":           {"first": "Eugenio",    "last": "Suarez"},
    # F
    "Fernando Tatis Jr.":       {"first": "Fernando",   "last": "Tatis"},
    "Freddie Freeman":          {"first": "Freddie",    "last": "Freeman"},
    # G
    "George Springer":          {"first": "George",     "last": "Springer"},
    "Giancarlo Stanton":        {"first": "Giancarlo",  "last": "Stanton"},
    "Gleyber Torres":           {"first": "Gleyber",    "last": "Torres"},
    "Gunnar Henderson":         {"first": "Gunnar",     "last": "Henderson"},
    # H
    "Ha-Seong Kim":             {"first": "Ha-Seong",   "last": "Kim"},
    "Hunter Renfroe":           {"first": "Hunter",     "last": "Renfroe"},
    # I
    "Ian Happ":                 {"first": "Ian",        "last": "Happ"},
    "Isaac Paredes":            {"first": "Isaac",      "last": "Paredes"},
    "Isiah Kiner-Falefa":       {"first": "Isiah",      "last": "Kiner-Falefa"},
    # J
    "J.D. Martinez":            {"first": "J.D.",       "last": "Martinez"},
    "Jazz Chisholm Jr.":        {"first": "Jazz",       "last": "Chisholm"},
    "Jeff McNeil":              {"first": "Jeff",       "last": "McNeil"},
    "Jeremy Pena":              {"first": "Jeremy",     "last": "Pena"},
    "Jorge Soler":              {"first": "Jorge",      "last": "Soler"},
    "Jose Abreu":               {"first": "Jose",       "last": "Abreu"},
    "Jose Altuve":              {"first": "Jose",       "last": "Altuve"},
    "Jose Ramirez":             {"first": "Jose",       "last": "Ramirez"},
    "Juan Soto":                {"first": "Juan",       "last": "Soto"},
    "Julio Rodriguez":          {"first": "Julio",      "last": "Rodriguez"},
    # K
    "Ketel Marte":              {"first": "Ketel",      "last": "Marte"},
    "Kris Bryant":              {"first": "Kris",       "last": "Bryant"},
    "Kyle Schwarber":           {"first": "Kyle",       "last": "Schwarber"},
    "Kyle Tucker":              {"first": "Kyle",       "last": "Tucker"},
    # L
    "Luis Arraez":              {"first": "Luis",       "last": "Arraez"},
    "Luis Robert":              {"first": "Luis",       "last": "Robert"},
    # M
    "Manny Machado":            {"first": "Manny",      "last": "Machado"},
    "Marcus Semien":            {"first": "Marcus",     "last": "Semien"},
    "Matt Olson":               {"first": "Matt",       "last": "Olson"},
    "Michael Harris II":        {"first": "Michael",    "last": "Harris"},
    "Mike Trout":               {"first": "Mike",       "last": "Trout"},
    "Mookie Betts":             {"first": "Mookie",     "last": "Betts"},
    # N
    "Nathaniel Lowe":           {"first": "Nathaniel",  "last": "Lowe"},
    "Nick Castellanos":         {"first": "Nick",       "last": "Castellanos"},
    "Nico Hoerner":             {"first": "Nico",       "last": "Hoerner"},
    "Nolan Arenado":            {"first": "Nolan",      "last": "Arenado"},
    "Nolan Gorman":             {"first": "Nolan",      "last": "Gorman"},
    # O
    "Ozzie Albies":             {"first": "Ozzie",      "last": "Albies"},
    # P
    "Paul Goldschmidt":         {"first": "Paul",       "last": "Goldschmidt"},
    "Paul Skenes":              {"first": "Paul",       "last": "Skenes"},
    "Pete Alonso":              {"first": "Pete",       "last": "Alonso"},
    # R
    "Rafael Devers":            {"first": "Rafael",     "last": "Devers"},
    "Randy Arozarena":          {"first": "Randy",      "last": "Arozarena"},
    "Ronald Acuna Jr.":         {"first": "Ronald",     "last": "Acuna"},
    # S
    "Salvador Perez":           {"first": "Salvador",   "last": "Perez"},
    "Seiya Suzuki":             {"first": "Seiya",      "last": "Suzuki"},
    "Shohei Ohtani":            {"first": "Shohei",     "last": "Ohtani"},
    "Spencer Torkelson":        {"first": "Spencer",    "last": "Torkelson"},
    "Steven Kwan":              {"first": "Steven",     "last": "Kwan"},
    # T
    "Teoscar Hernandez":        {"first": "Teoscar",    "last": "Hernandez"},
    "Tommy Edman":              {"first": "Tommy",      "last": "Edman"},
    "Trea Turner":              {"first": "Trea",       "last": "Turner"},
    "Trey Mancini":             {"first": "Trey",       "last": "Mancini"},
    "Tyler O'Neill":            {"first": "Tyler",      "last": "O'Neill"},
    # V
    "Vladimir Guerrero Jr.":    {"first": "Vladimir",   "last": "Guerrero"},
    # W
    "Willy Adames":             {"first": "Willy",      "last": "Adames"},
    "Xander Bogaerts":          {"first": "Xander",     "last": "Bogaerts"},
    # Y
    "Yandy Diaz":               {"first": "Yandy",      "last": "Diaz"},
    "Yordan Alvarez":           {"first": "Yordan",     "last": "Alvarez"},
}

SEASONS = [2021, 2022, 2023, 2024, 2025]

SWING_EVENTS = {
    "hit_into_play", "swinging_strike", "swinging_strike_blocked",
    "foul", "foul_tip", "foul_bunt", "missed_bunt",
}

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
    # Lower bound drops 1°/mph; upper bound rises 20/18°/mph. Both cap at 116 mph.
    ev = df["launch_speed"].to_numpy(dtype=float, na_value=np.nan)
    la = df["launch_angle"].to_numpy(dtype=float, na_value=np.nan)
    ev_cap   = np.clip(ev, 0, 116)
    delta    = ev_cap - 98
    min_la   = 26 - delta                   # 26° → 8°
    max_la   = 30 + delta * (20.0 / 18.0)   # 30° → 50°

    df["is_barrel"] = (
        (~np.isnan(ev)) & (~np.isnan(la)) &
        (ev >= 98) &
        (la >= min_la) & (la <= max_la)
    )
    return df

def build_barrel_kde(barrel_df: pd.DataFrame, xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """2-D Gaussian KDE of barrel pitch locations; returned grid is normalized 0-1."""
    x = barrel_df["plate_x"].values
    y = barrel_df["plate_z"].values
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
    """
    For every pitch, interpolate its barrel-KDE density and return a boolean
    Series: True when density >= threshold (i.e. inside the player's hot zone).
    """
    if kde_grid.max() == 0:
        return pd.Series(False, index=pitch_df.index)

    interp = RegularGridInterpolator(
        (yc, xc), kde_grid,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    pts = pitch_df[["plate_z", "plate_x"]].values
    density = interp(pts)
    return pd.Series(density >= threshold, index=pitch_df.index)

def binned_swing_rate(df: pd.DataFrame,
                      xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """Bin swing-rate on a grid (requires 'plate_x', 'plate_z', 'is_swing')."""
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

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("⚾ Controls")
player_name = st.sidebar.selectbox("Player", list(PLAYERS.keys()), index=0)
season      = st.sidebar.selectbox("Season", SEASONS[::-1], index=0)
overlay     = st.sidebar.radio(
    "Heatmap overlay",
    ["Barrels only", "Swing % (in barrel zone)", "Swing % (outside barrel zone)"],
    index=0,
)
kde_threshold = st.sidebar.slider(
    "Barrel KDE hot-zone threshold",
    min_value=0.10, max_value=0.80, value=0.40, step=0.05,
    help="Fraction of peak barrel density a pitch must exceed to be counted as 'inside' the barrel hot zone.",
)
colorscale = st.sidebar.selectbox("Color scale", ["Hot", "Viridis", "Plasma", "RdYlBu_r"], index=0)

# ── Load & classify ────────────────────────────────────────────────────────────
st.title(f"🔥 {player_name} — {season} Statcast Dashboard")

with st.spinner(f"Loading Statcast data for {player_name} ({season})…"):
    info   = PLAYERS[player_name]
    raw_df = load_statcast(info["first"], info["last"], season)

if raw_df is None or raw_df.empty:
    st.error("No Statcast data found for this player/season combination.")
    st.stop()

df        = classify(raw_df)
pitch_df  = df.dropna(subset=["plate_x", "plate_z"])
# Balls actually put in play (type == 'X') — correct denominator for Barrel%
batted_df = df[df["type"] == "X"].dropna(subset=["launch_speed", "launch_angle"])
barrel_df = batted_df[batted_df["is_barrel"]]

# Grid for heatmap
xi = np.linspace(-1.5, 1.5, 40)
yi = np.linspace(0.5,  5.0, 40)
xc = (xi[:-1] + xi[1:]) / 2   # bin centres
yc = (yi[:-1] + yi[1:]) / 2

# ── Build barrel KDE, then classify every *pitch* against it ──────────────────
kde_grid = build_barrel_kde(barrel_df, xc, yc)    # shape (39, 39), normalized 0-1

in_hot_zone  = in_barrel_kde_zone(pitch_df, kde_grid, xc, yc, kde_threshold)
hot_pitch_df = pitch_df[in_hot_zone]
cold_pitch_df = pitch_df[~in_hot_zone]

# ── Summary metrics (all relative to the barrel KDE map, not generic zone) ────
batted_balls  = len(batted_df)
total_swings  = pitch_df["is_swing"].sum()

barrel_rate        = len(barrel_df) / max(batted_balls, 1) * 100
barrel_zone_swing  = hot_pitch_df["is_swing"].mean()  * 100 if len(hot_pitch_df)  else 0.0
non_barrel_swing   = cold_pitch_df["is_swing"].mean() * 100 if len(cold_pitch_df) else 0.0
contact_rate       = (
    pitch_df[pitch_df["description"].isin({"hit_into_play", "foul", "foul_tip", "foul_bunt"})]
    .shape[0] / max(total_swings, 1) * 100
)

col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    (col1, f"{barrel_rate:.1f}%",       "Barrel Rate"),
    (col2, f"{barrel_zone_swing:.1f}%", "Swing % in Barrel Zone"),
    (col3, f"{non_barrel_swing:.1f}%",  "Swing % Outside Barrel Zone"),
    (col4, f"{contact_rate:.1f}%",      "Contact Rate"),
    (col5, str(len(barrel_df)),         "Total Barrels"),
]
for col, val, label in metrics:
    with col:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Main heatmap ───────────────────────────────────────────────────────────────
if overlay == "Barrels only":
    z_grid         = kde_grid
    title_overlay  = "Barrel KDE Density"
    colorbar_title = "Relative Density"
    tick_fmt       = ""

elif overlay == "Swing % (in barrel zone)":
    # Show swing rate only for pitches that landed inside the barrel hot zone
    z_grid         = binned_swing_rate(hot_pitch_df, xi, yi)
    title_overlay  = f"Swing % — inside barrel hot zone (threshold ≥ {kde_threshold:.0%})"
    colorbar_title = "Swing %"
    tick_fmt       = ".0%"

else:  # outside barrel zone
    z_grid         = binned_swing_rate(cold_pitch_df, xi, yi)
    title_overlay  = f"Swing % — outside barrel hot zone (threshold < {kde_threshold:.0%})"
    colorbar_title = "Swing %"
    tick_fmt       = ".0%"

fig = go.Figure()

fig.add_trace(go.Heatmap(
    x=xc, y=yc, z=z_grid,
    colorscale=colorscale,
    colorbar=dict(title=colorbar_title, tickformat=tick_fmt),
    opacity=0.85,
    zmin=0,
))

# KDE contour overlay showing the hot-zone boundary (always visible)
if kde_grid.max() > 0:
    fig.add_trace(go.Contour(
        x=xc, y=yc, z=kde_grid,
        contours=dict(
            start=kde_threshold, end=kde_threshold, size=0,
            coloring="none",
        ),
        line=dict(color="#a6e3a1", width=2, dash="dot"),
        showscale=False,
        name=f"Hot zone boundary ({kde_threshold:.0%})",
        hoverinfo="skip",
    ))

# Barrel scatter dots
if len(barrel_df) > 0:
    fig.add_trace(go.Scatter(
        x=barrel_df["plate_x"],
        y=barrel_df["plate_z"],
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
    title=dict(
        text=f"{player_name} · {season} · {title_overlay}",
        font=dict(size=16, color="white"),
    ),
    xaxis=dict(
        title="Horizontal Position (ft) — Pitcher's View",
        range=[1.5, -1.5], zeroline=False, color="white", gridcolor="#313244",
    ),
    yaxis=dict(
        title="Vertical Position (ft)",
        range=[0.5, 5.0], zeroline=False, color="white", gridcolor="#313244",
        scaleanchor="x", scaleratio=1,
    ),
    paper_bgcolor="#1e1e2e",
    plot_bgcolor="#181825",
    font=dict(color="white"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
    height=600,
    margin=dict(l=60, r=40, t=70, b=60),
)

fig.add_shape(
    type="path",
    path="M -0.83 0.6 L 0.83 0.6 L 0.83 0.75 L 0 0.9 L -0.83 0.75 Z",
    fillcolor="rgba(255,255,255,0.08)", line=dict(color="#a6adc8", width=1),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Green dotted contour = barrel KDE hot-zone boundary at {kde_threshold:.0%} of peak density · "
    f"Red dots = individual barrels · Dashed white rectangle = standard strike zone"
)

# ── Distribution panels ────────────────────────────────────────────────────────
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
                         annotation_text="Barrel LA band", annotation_position="top left")
        fig_la.update_layout(
            barmode="overlay", title="Launch Angle Distribution",
            xaxis_title="Launch Angle (°)", yaxis_title="Count",
            paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
            font=dict(color="white"), legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=320, margin=dict(l=50, r=20, t=50, b=50),
        )
        st.plotly_chart(fig_la, use_container_width=True)

# ── Spray chart ────────────────────────────────────────────────────────────────
st.subheader("Spray Chart")
if "hc_x" in df.columns and "hc_y" in df.columns:
    spray_df = batted_df.dropna(subset=["hc_x", "hc_y"])
    if not spray_df.empty:
        fig_spray = go.Figure()
        non_barrel  = spray_df[~spray_df["is_barrel"]]
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
            hovertemplate="Exit Velo: %{customdata[0]:.1f} mph<br>LA: %{customdata[1]:.1f}°<br>Result: %{customdata[2]}<extra></extra>",
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

# ── Raw data expander ──────────────────────────────────────────────────────────
with st.expander("Raw barrel data"):
    cols = ["game_date", "pitch_type", "plate_x", "plate_z",
            "launch_speed", "launch_angle", "hit_distance_sc", "events", "description"]
    available = [c for c in cols if c in barrel_df.columns]
    st.dataframe(
        barrel_df[available].sort_values("game_date", ascending=False),
        use_container_width=True, height=300,
    )

st.caption("Data: pybaseball / Baseball Savant · Statcast · Pitcher's perspective")
