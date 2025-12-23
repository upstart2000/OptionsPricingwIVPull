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
    # The math expects decimals (e.g., 0.043), not percentages (4.3)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_val**2) * T_val) / (sigma_val * np.sqrt(T_val))
    d2 = d1 - sigma_val * np.sqrt(T_val)
    if opt_type == "Call":
        return (S * np.exp(-q * T_val) * norm.cdf(d1)) - (K * np.exp(-r * T_val) * norm.cdf(d2))
    else:
        return (K * np.exp(-r * T_val) * norm.cdf(-d2)) - (S * np.exp(-q * T_val) * norm.cdf(-d1))

def find_implied_vol(market_price, S, K, T, r, q, opt_type):
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, q, opt_type) - market_price
    try:
        return brentq(objective, 1e-5, 5.0)
    except:
        return None

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

# --- 2. INITIALIZE SESSION STATE ---
if 'ticker_price' not in st.session_state: st.session_state['ticker_price'] = 150.0
if 'div_yield_pct' not in st.session_state: st.session_state['div_yield_pct'] = 0.00
if 'last_ticker' not in st.session_state: st.session_state['last_ticker'] = ""
if 'solved_iv' not in st.session_state: st.session_state['solved_iv'] = 0.40

# --- 3. UI SETUP ---
st.set_page_config(page_title="Professional Options Solver", layout="wide")
st.title("🛡️ Professional IV Solver & Greeks")

with st.sidebar:
    st.header("1. Live Data Feed")
    ticker_sym = st.text_input("Ticker Symbol", value="NVDA").upper()
    ticker_obj = yf.Ticker(ticker_sym)

    # --- AUTO-LOAD ON TICKER CHANGE ---
    if ticker_sym != st.session_state['last_ticker']:
        try:
            info = ticker_obj.info
            st.session_state['ticker_price'] = info.get('regularMarketPrice', info.get('currentPrice', 150.0))
            
            # Yahoo returns yield as 0.0002 for 0.02%. We multiply by 100 to show '0.02' in the box.
            raw_div = info.get('dividendYield', 0.0) if info.get('dividendYield') else 0.0
            st.session_state['div_yield_pct'] = raw_div * 100
            
            st.session_state['last_ticker'] = ticker_sym
        except:
            st.error("Could not fetch ticker info.")

    # --- REFRESH PRICE ONLY ---
    c1, c2 = st.columns([3, 1])
    c1.metric("Spot Price", f"${st.session_state['ticker_price']:.2f}")
    if c2.button("🔄"):
        try:
            st.session_state['ticker_price'] = ticker_obj.fast_info['lastPrice']
        except:
            st.error("Refresh failed.")

    st.header("2. Option Setup")
    opt_type = st.radio("Option Type", ["Call", "Put"])
    target_k = st.number_input("Strike Price ($)", value=float(np.round(st.session_state['ticker_price'], 0)))
    
    try:
        all_exps = ticker_obj.options
        selected_exp = st.selectbox("Select Expiration", all_exps)
        expiry_dt = date.fromisoformat(selected_exp)
        days_to_exp = (expiry_dt - date.today()).days
        T = max(days_to_exp, 1) / 365.0
    except:
        T = 30/365.0

    st.header("3. Parameters (%)")
    # Risk-Free: Enter 4.3 for 4.3%
    r_input = st.number_input("Risk-Free Rate (%)", value=4.300, format="%.3f")
    
    # Div Yield: Enter 0.02 for 0.02% or 6.8 for 6.8%
    q_input = st.number_input("Dividend Yield (%)", value=float(st.session_state['div_yield_pct']), format="%.4f")
    
    # Internal Conversion for Black-Scholes: Percentage to Decimal (4.3 -> 0.043)
    r_decimal = r_input / 100.0
    q_decimal = q_input / 100.0
    
    strike_step = st.number_input("Strike Interval ($)", value=5.0)

    st.header("4. IV Solver")
    if st.button("⚡ Solve IV from Market"):
        try:
            chain = ticker_obj.option_chain(selected_exp)
            df = chain.calls if opt_type == "Call" else chain.puts
            row = df.iloc[(df['strike'] - target_k).abs().idxmin()]
            mid = (row['bid'] + row['ask']) / 2
            
            # Solves using the decimal versions
            solved = find_implied_vol(mid, st.session_state['ticker_price'], target_k, T, r_decimal, q_decimal, opt_type)
            if solved:
                st.session_state['solved_iv'] = solved
                st.success(f"IV Found: {solved*100:.2f}%")
        except:
            st.error("Option fetch failed.")

    final_iv = st.number_input("Final IV (Decimal, e.g. 0.45)", value=float(st.session_state['solved_iv']), format="%.4f")

# --- 4. RESULTS DISPLAY ---
# Logic remains consistent using r_decimal and q_decimal
strikes_to_show = [target_k + (i * strike_step) for i in range(-5, 6)]
pricing_data, greeks_data = [], []

for k in strikes_to_show:
    p_main = black_scholes_price(st.session_state['ticker_price'], k, T, r_decimal, final_iv, q_decimal, opt_type)
    pricing_data.append({"Strike": f"${k:,.2f}", f"Price at {final_iv*100:.2f}% IV": f"${p_main:.2f}"})
    
    d, g, th = get_greeks(st.session_state['ticker_price'], k, T, r_decimal, final_iv, q_decimal, opt_type)
    greeks_data.append({
        "Strike": f"${k:,.2f}", 
        "Delta": f"{d:.4f}", 
        "Gamma": f"{g:.4f}", 
        "Theta (Daily)": f"${th:.4f}"
    })

t1, t2 = st.tabs(["💰 Pricing Matrix", "📈 Greeks Matrix"])
with t1: st.table(pd.DataFrame(pricing_data))
with t2: st.table(pd.DataFrame(greeks_data))
