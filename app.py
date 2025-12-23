import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import date

# --- 1. CORE MATH FUNCTIONS ---
def black_scholes_price(S, K, T, r, sigma, q, opt_type="Call"):
    T_val = max(T, 1e-10)
    sigma_val = max(sigma, 1e-4)
    # Math expects decimals (4.3% -> 0.043)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_val**2) * T_val) / (sigma_val * np.sqrt(T_val))
    d2 = d1 - sigma_val * np.sqrt(T_val)
    if opt_type == "Call":
        return (S * np.exp(-q * T_val) * norm.cdf(d1)) - (K * np.exp(-r * T_val) * norm.cdf(d2))
    else:
        return (K * np.exp(-r * T_val) * norm.cdf(-d2)) - (S * np.exp(-q * T_val) * norm.cdf(-d1))

def get_greeks(S, K, T, r, sigma, q, opt_type="Call"):
    T_val = max(T, 1e-10)
    sigma_val = max(sigma, 1e-4)
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

# --- 2. SESSION STATE INITIALIZATION ---
if 'last_ticker' not in st.session_state: st.session_state['last_ticker'] = ""
if 'ticker_price' not in st.session_state: st.session_state['ticker_price'] = 100.0
if 'manual_div_pct' not in st.session_state: st.session_state['manual_div_pct'] = 0.0
if 'solved_iv' not in st.session_state: st.session_state['solved_iv'] = 0.40

# --- 3. UI SIDEBAR ---
st.set_page_config(page_title="Options Solver", layout="wide")
st.title("🛡️ Professional IV Solver & Greeks")

with st.sidebar:
    st.header("1. Data Feed")
    ticker_sym = st.text_input("Ticker", value="NVDA").upper()
    ticker_obj = yf.Ticker(ticker_sym)

    # FETCH DATA ON TICKER CHANGE
    if ticker_sym != st.session_state['last_ticker']:
        try:
            info = ticker_obj.info
            # 1. Price
            st.session_state['ticker_price'] = info.get('regularMarketPrice', info.get('currentPrice', 100.0))
            # 2. Dividend Logic: Yahoo (0.0002) -> Box (0.02)
            raw_yield = info.get('dividendYield', 0.0) if info.get('dividendYield') else 0.0
            st.session_state['manual_div_pct'] = float(raw_yield * 100)
            
            st.session_state['last_ticker'] = ticker_sym
        except:
            st.error("Ticker fetch error.")

    # REFRESH PRICE ONLY (Does NOT touch Dividend)
    col1, col2 = st.columns([3, 1])
    col1.metric("Spot Price", f"${st.session_state['ticker_price']:.2f}")
    if col2.button("🔄"):
        st.session_state['ticker_price'] = ticker_obj.fast_info['lastPrice']

    st.header("2. Parameters (%)")
    # User enters 4.3 meaning 4.3%
    r_in = st.number_input("Risk-Free Rate (%)", value=4.3, format="%.3f")
    # User enters 0.02 meaning 0.02%
    q_in = st.number_input("Dividend Yield (%)", value=st.session_state['manual_div_pct'], format="%.4f")
    
    # CONVERT TO DECIMAL FOR MATH
    r_math = r_in / 100.0
    q_math = q_in / 100.0

    st.header("3. Option Config")
    opt_type = st.radio("Type", ["Call", "Put"])
    target_k = st.number_input("Strike ($)", value=float(np.round(st.session_state['ticker_price'], 0)))
    
    try:
        exps = ticker_obj.options
        sel_exp = st.selectbox("Expiry", exps)
        days = (date.fromisoformat(sel_exp) - date.today()).days
        T = max(days, 1) / 365.0
    except:
        T = 30/365.0

    if st.button("⚡ Solve IV"):
        try:
            chain = ticker_obj.option_chain(sel_exp)
            df = chain.calls if opt_type == "Call" else chain.puts
            mid = (df.iloc[(df['strike'] - target_k).abs().idxmin()][['bid', 'ask']].mean())
            
            def obj(s): return black_scholes_price(st.session_state['ticker_price'], target_k, T, r_math, s, q_math, opt_type) - mid
            st.session_state['solved_iv'] = brentq(obj, 1e-5, 5.0)
            st.success(f"IV: {st.session_state['solved_iv']*100:.2f}%")
        except:
            st.error("IV Solve failed.")

    f_iv = st.number_input("Final IV (Decimal)", value=float(st.session_state['solved_iv']), format="%.4f")
    step = st.number_input("Strike Step", value=5.0)

# --- 4. OUTPUT ---
strikes = [target_k + (i * step) for i in range(-5, 6)]
p_list, g_list = [], []

for k in strikes:
    px = black_scholes_price(st.session_state['ticker_price'], k, T, r_math, f_iv, q_math, opt_type)
    p_list.append({"Strike": f"${k:.2f}", f"Price ({f_iv*100:.1f}% IV)": f"${px:.2f}"})
    
    de, ga, th = get_greeks(st.session_state['ticker_price'], k, T, r_math, f_iv, q_math, opt_type)
    g_list.append({"Strike": f"${k:.2f}", "Delta": f"{de:.4f}", "Gamma": f"{ga:.4f}", "Theta": f"{th:.4f}"})

st.tabs(["💰 Price", "📈 Greeks"])[0].table(pd.DataFrame(p_list))
st.tabs(["💰 Price", "📈 Greeks"])[1].table(pd.DataFrame(g_list))
