import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import date

# --- 1. CORE MATH FUNCTIONS ---
def black_scholes_price(S, K, T, r, sigma, q, opt_type="Call"):
    T_val = max(T, 1e-9)
    sigma_val = max(sigma, 1e-9)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_val**2) * T_val) / (sigma_val * np.sqrt(T_val))
    d2 = d1 - sigma_val * np.sqrt(T_val)
    
    if opt_type == "Call":
        price = (S * np.exp(-q * T_val) * norm.cdf(d1)) - (K * np.exp(-r * T_val) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T_val) * norm.cdf(-d2)) - (S * np.exp(-q * T_val) * norm.cdf(-d1))
    return max(price, 0.0)

def get_greeks(S, K, T, r, sigma, q, opt_type="Call"):
    T_val = max(T, 1e-9)
    sigma_val = max(sigma, 1e-9)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_val**2) * T_val) / (sigma_val * np.sqrt(T_val))
    d2 = d1 - sigma_val * np.sqrt(T_val)
    
    delta = np.exp(-q * T_val) * (norm.cdf(d1) if opt_type == "Call" else norm.cdf(d1) - 1)
    gamma = (norm.pdf(d1) * np.exp(-q * T_val)) / (S * sigma_val * np.sqrt(T_val))
    
    term1 = -(S * sigma_val * np.exp(-q * T_val) * norm.pdf(d1)) / (2 * np.sqrt(T_val))
    if opt_type == "Call":
        theta = (term1 + (q * S * np.exp(-q * T_val) * norm.cdf(d1)) - (r * K * np.exp(-r * T_val) * norm.cdf(d2))) / 365.0
    else:
        theta = (term1 - (q * S * np.exp(-q * T_val) * norm.cdf(-d1)) + (r * K * np.exp(-r * T_val) * norm.cdf(-d2))) / 365.0
    return delta, gamma, theta

# --- 2. SESSION STATE ---
if 'last_ticker' not in st.session_state: st.session_state['last_ticker'] = ""
if 'ticker_price' not in st.session_state: st.session_state['ticker_price'] = 10.0
if 'manual_div_pct' not in st.session_state: st.session_state['manual_div_pct'] = 0.0
if 'solved_iv' not in st.session_state: st.session_state['solved_iv'] = 0.40

# --- 3. UI SIDEBAR ---
st.set_page_config(page_title="Options Solver", layout="wide")
st.sidebar.header("1. Data Feed")

ticker_sym = st.sidebar.text_input("Ticker Symbol", value="MFA").upper()
ticker_obj = yf.Ticker(ticker_sym)

# Load data on ticker change
if ticker_sym != st.session_state['last_ticker']:
    try:
        info = ticker_obj.info
        st.session_state['ticker_price'] = info.get('regularMarketPrice', info.get('currentPrice', 10.0))
        st.session_state['manual_div_pct'] = info.get('dividendYield', 0.0) or 0.0
        st.session_state['last_ticker'] = ticker_sym
    except:
        st.sidebar.error("Fetch failed.")

# --- THE REFRESH SECTION ---
c1, c2 = st.sidebar.columns([3, 1])
c1.metric("Spot Price", f"${st.session_state['ticker_price']:.2f}")
if c2.button("🔄"):
    try:
        # Fetching fresh price without touching other inputs
        fresh_info = ticker_obj.fast_info
        st.session_state['ticker_price'] = fresh_info['lastPrice']
        st.toast(f"Updated {ticker_sym} price!")
    except:
        st.sidebar.error("Price refresh failed.")

st.sidebar.markdown("---")
st.sidebar.header("2. Parameters (%)")
r_pct = st.sidebar.number_input("Risk-Free Rate (%)", value=4.30, format="%.2f")
q_pct = st.sidebar.number_input("Dividend Yield (%)", value=float(st.session_state['manual_div_pct']), format="%.2f")
r, q = r_pct/100, q_pct/100

st.sidebar.header("3. Option Setup")
opt_type = st.sidebar.radio("Type", ["Call", "Put"])
target_k = st.sidebar.number_input("Strike Price", value=float(np.round(st.session_state['ticker_price'], 0)))

try:
    exps = ticker_obj.options
    selected_exp = st.sidebar.selectbox("Expiration", exps)
    T = (date.fromisoformat(selected_exp) - date.today()).days / 365.0
except:
    T = 0.1

if st.sidebar.button("⚡ Solve Implied Vol"):
    try:
        chain = ticker_obj.option_chain(selected_exp)
        df = chain.calls if opt_type == "Call" else chain.puts
        mid = df.iloc[(df['strike'] - target_k).abs().idxmin()][['bid', 'ask']].mean()
        def f(s): return black_scholes_price(st.session_state['ticker_price'], target_k, T, r, s, q, opt_type) - mid
        st.session_state['solved_iv'] = brentq(f, 1e-6, 5.0)
    except:
        st.sidebar.error("IV Solver failed.")

final_iv = st.sidebar.number_input("Volatility (Decimal)", value=float(st.session_state['solved_iv']), format="%.4f")
step = st.sidebar.number_input("Strike Interval", value=1.0 if st.session_state['ticker_price'] < 25 else 5.0)

# --- 4. MAIN DISPLAY ---
st.header(f"Analysis: {ticker_sym} | Spot: ${st.session_state['ticker_price']:.2f}")

strikes = [target_k + (i * step) for i in range(-5, 6)]
p_data, g_data = [], []

for k in strikes:
    if k <= 0: continue
    price = black_scholes_price(st.session_state['ticker_price'], k, T, r, final_iv, q, opt_type)
    de, ga, th = get_greeks(st.session_state['ticker_price'], k, T, r, final_iv, q, opt_type)
    p_data.append({"Strike": f"${k:.2f}", "Theoretical Price": f"${price:.2f}"})
    g_data.append({"Strike": f"${k:.2f}", "Delta": f"{de:.4f}", "Gamma": f"{ga:.4f}", "Theta": f"{th:.4f}"})

t1, t2 = st.tabs(["💰 Price Matrix", "📈 Greeks Matrix"])
t1.table(pd.DataFrame(p_data))
t2.table(pd.DataFrame(g_data))
