import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import requests

# Set page configuration
st.set_page_config(
    page_title="NHL Betting Projection & +EV Value Finder",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1. DATA FETCHING & CACHING
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_nhl_standings():
    """Fetches team goals scored/against per game from official NHL API."""
    try:
        url = "https://api-web.nhle.com/v1/standings/now"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            teams = []
            for team in data.get('standings', []):
                tricode = team['teamAbbrev']['default']
                games = team['gamesPlayed']
                gf = team['goalFor']
                ga = team['goalAgainst']
                if games > 0:
                    teams.append({
                        'team': tricode,
                        'gf_pg': gf / games,
                        'ga_pg': ga / games
                    })
            df = pd.DataFrame(teams).set_index('team').sort_index()
            return df, "Live NHL API Data"
    except Exception:
        pass

    # Fallback Data if API is unavailable
    teams = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", 
             "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", 
             "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WPG", "WSH"]
    fallback_df = pd.DataFrame([{'team': t, 'gf_pg': 3.10, 'ga_pg': 3.10} for t in teams]).set_index('team')
    return fallback_df, "Offline Baseline Data"

# Load Standings Data
df_stats, data_source = fetch_nhl_standings()

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def prob_to_american(p):
    if p >= 0.5:
        return int(round(-100 * (p / (1 - p))))
    else:
        return int(round(100 * ((1 - p) / p)))

def calc_ev(p, american_odds):
    if american_odds < 0:
        profit = 100 / (-american_odds / 100)
    else:
        profit = american_odds
    return (p * profit) - ((1 - p) * 100)

def kelly_criterion(p, american_odds, fraction=0.25):
    if american_odds > 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / abs(american_odds)
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f * fraction * 100)

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & INPUTS
# -----------------------------------------------------------------------------
st.sidebar.header("🏒 Matchup & Market Inputs")

team_list = list(df_stats.index)
away_default = team_list.index("TOR") if "TOR" in team_list else 0
home_default = team_list.index("BOS") if "BOS" in team_list else min(1, len(team_list)-1)

away_team = st.sidebar.selectbox("Away Team", team_list, index=away_default)
home_team = st.sidebar.selectbox("Home Team", team_list, index=home_default)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Parameters")
hia_pct = st.sidebar.slider("Home Ice Advantage (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Sportsbook Market Odds")
col_odds1, col_odds2 = st.sidebar.columns(2)
with col_odds1:
    away_market_odds = st.number_input(f"{away_team} Odds", value=+115, step=5)
with col_odds2:
    home_market_odds = st.number_input(f"{home_team} Odds", value=-135, step=5)

st.sidebar.caption(f"Data Status: **{data_source}**")

# Validation
if away_team == home_team:
    st.error("Please select two different teams for Away and Home.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. PROJECTION ENGINE COMPUTATION
# -----------------------------------------------------------------------------
hia_mult = 1.0 + (hia_pct / 100.0)
lg_avg_gf = df_stats['gf_pg'].mean()

away_stat = df_stats.loc[away_team]
home_stat = df_stats.loc[home_team]

lambda_home = (home_stat['gf_pg'] * away_stat['ga_pg'] / lg_avg_gf) * hia_mult
lambda_away = (away_stat['gf_pg'] * home_stat['ga_pg'] / lg_avg_gf)

# 10x10 Poisson Matrix
goals = np.arange(0, 10)
p_home = stats.poisson.pmf(goals, lambda_home)
p_away = stats.poisson.pmf(goals, lambda_away)
matrix = np.outer(p_home, p_away)

p_home_reg = np.sum(np.tril(matrix, -1))
p_away_reg = np.sum(np.triu(matrix, 1))
p_tie = np.sum(np.diag(matrix))

p_home_win = p_home_reg + (0.5 * p_tie)
p_away_win = p_away_reg + (0.5 * p_tie)

ev_away = calc_ev(p_away_win, away_market_odds)
ev_home = calc_ev(p_home_win, home_market_odds)

kelly_away = kelly_criterion(p_away_win, away_market_odds)
kelly_home = kelly_criterion(p_home_win, home_market_odds)

# -----------------------------------------------------------------------------
# 5. MAIN DASHBOARD DISPLAY
# -----------------------------------------------------------------------------
st.title("🏒 NHL Betting Projection & +EV Value Finder")
st.markdown(f"### Matchup Breakdown: **{away_team}** @ **{home_team}**")

# Top Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Projected Total Goals", f"{lambda_away + lambda_home:.2f}", delta=f"{lambda_away:.2f} - {lambda_home:.2f}")
col2.metric(f"{away_team} Win Prob", f"{p_away_win*100:.1f}%", delta=f"Fair Odds: {prob_to_american(p_away_win):+d}")
col3.metric(f"{home_team} Win Prob", f"{p_home_win*100:.1f}%", delta=f"Fair Odds: {prob_to_american(p_home_win):+d}")
col4.metric("OT/Shootout Prob", f"{p_tie*100:.1f}%")

st.markdown("---")

# Projection & +EV Table Summary
st.subheader("📈 Moneyline & +EV Edge Analysis")

summary_df = pd.DataFrame([
    {
        "Team": f"{away_team} (Away)",
        "Exp Goals (λ)": round(lambda_away, 2),
        "Model Win %": f"{p_away_win*100:.1f}%",
        "Fair Odds": f"{prob_to_american(p_away_win):+d}",
        "Market Odds": f"{away_market_odds:+d}",
        "Expected Value (+EV)": f"{ev_away:+.2f}%",
        "Kelly Stake (1/4)": f"{kelly_away:.2f}%"
    },
    {
        "Team": f"{home_team} (Home)",
        "Exp Goals (λ)": round(lambda_home, 2),
        "Model Win %": f"{p_home_win*100:.1f}%",
        "Fair Odds": f"{prob_to_american(p_home_win):+d}",
        "Market Odds": f"{home_market_odds:+d}",
        "Expected Value (+EV)": f"{ev_home:+.2f}%",
        "Kelly Stake (1/4)": f"{kelly_home:.2f}%"
    }
])

st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Value Alert Boxes
if ev_away > 0:
    st.success(f"🎯 **VALUE FOUND!** {away_team} has a **{ev_away:+.2f}% Expected Value** at market odds of {away_market_odds:+d}. Recommended Stake: **{kelly_away:.2f}% of bankroll**.")
if ev_home > 0:
    st.success(f"🎯 **VALUE FOUND!** {home_team} has a **{ev_home:+.2f}% Expected Value** at market odds of {home_market_odds:+d}. Recommended Stake: **{kelly_home:.2f}% of bankroll**.")
if ev_away <= 0 and ev_home <= 0:
    st.info("ℹ️️ No positive expected value (+EV) edge found at the given sportsbook lines.")

st.markdown("---")

# Score Matrix Display
st.subheader("📊 Score Probability Matrix (10x10 Poisson Distribution)")

matrix_df = pd.DataFrame(
    matrix * 100,
    index=[f"Home {i}" for i in range(10)],
    columns=[f"Away {j}" for j in range(10)]
)

st.dataframe(matrix_df.style.format("{:.2f}%").background_gradient(cmap="Blues"), use_container_width=True)
streamlit>=1.25.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
requests>=2.28.0
