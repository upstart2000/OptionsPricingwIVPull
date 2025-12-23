
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
        # Range: 0.01% to 500% vol
        return brentq(objective, 1e-5, 5.0)
    except Exception:
        return None

def get_greeks(S, K, T, r, sigma, q, opt_type="Call"):
    T_val = max(T, 1e-10)
    sigma_val = max(sigma, 1e-4)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_val**2) * T_val) / (sigma_val * np.sqrt(T_val))
    d2 = d1 - sigma_val * np.sqrt(T_val)
    
    if opt_type == "Call":
        delta = np.exp(-q * T_val) * norm.cdf(d1)
    else:
        delta = np.exp(-q * T_val) * (norm.cdf(d1) - 1)
        
    gamma = (norm.pdf(d1) * np.exp(-q * T_val)) / (S * sigma_val * np.sqrt(T_val))
    
    term1 = -(S * sigma_val * np.exp(-q * T_val) * norm.pdf(d1)) / (2 * np.sqrt(T_val))
    if opt_type == "Call":
        theta = (term1 + (q * S * np.exp(-q * T_val) * norm.cdf(d1)) - (r * K * np.exp(-r * T_val) * norm.cdf(d2))) / 365.0
    else:
        theta = (term1 - (q * S * np.exp(-q * T_val) * norm.cdf(-d1)) + (r * K * np.exp(-r * T_val) * norm.cdf(-d2))) / 365.0
        
    return delta, gamma, theta

# --- 2. INITIALIZE SESSION STATE ---
# This prevents "KeyErrors" by ensuring variables exist before they are called
if 'solved_iv' not in st.session_state:
    st.session_state['solved_iv'] = 0.4000
if 'market_mid' not in st.session_state:
    st.session_state['market_mid'] = 0.0
if 'ticker_price' not in st.session_state:
    st.session_state['ticker_price'] = 150.0

# --- 3. UI SETUP ---
st.set_page_config(page_title="Options Master Tool", layout="wide")
st.title("🛡️ Professional IV Solver & Greeks")

with st.sidebar:
    st.header("1. Live Data Feed")
    c1, c2 = st.columns([3, 1])
    ticker_sym = c1.text_input("Ticker Symbol", value="AAPL").upper()
    
    # Simple price fetcher
    ticker_obj = yf.Ticker(ticker_sym)
    if c2.button("🔄"):
        try:
            st.session_state['ticker_price'] = ticker_obj.fast_info['lastPrice']
        except:
            st.error("Fetch failed.")
    
    current_price = st.session_state['ticker_price']
    st.metric("Spot Price", f"${current_price:.2f}")

    st.header("2. Option Setup")
    opt_type = st.radio("Option Type", ["Call", "Put"])
    target_k = st.number_input("Strike Price", value=float(np.round(current_price, 0)))
    
    try:
        all_exps = ticker_obj.options
        selected_exp = st.selectbox("Select Expiration", all_exps)
        expiry_dt = date.fromisoformat(selected_exp)
        days_to_exp = (expiry_dt - date.today()).days
        T = max(days_to_exp, 1) / 365.0
    except:
        st.error("Ticker or Expiry data unavailable.")
        T = 30/365.0

    st.header("3. Reverse IV Solver")
    if st.button("⚡ Solve IV from Market"):
        try:
            # Fetch midpoint
            chain = ticker_obj.option_chain(selected_exp)
            df = chain.calls if opt_type == "Call" else chain.puts
            row = df.iloc[(df['strike'] - target_k).abs().idxmin()]
            mid = (row['bid'] + row['ask']) / 2
            
            # Solve for IV
            solved = find_implied_vol(mid, current_price, target_k, T, 0.043, 0.0, opt_type)
            if solved:
                st.session_state['solved_iv'] = solved
                st.session_state['market_mid'] = mid
                st.success(f"IV Found: {solved*100:.2f}%")
            else:
                st.error("Could not find a math solution for this IV.")
        except:
            st.error("Option data not found for this strike/expiry.")

    final_iv = st.number_input("Implied Volatility (Manual Adjust)", value=float(st.session_state['solved_iv']), format="%.4f")
    strike_step = st.number_input("Strike Interval", value=5.0)

# --- 4. RESULTS DISPLAY ---
if st.session_state['market_mid'] > 0:
    st.info(f"Using Market Midpoint: **${st.session_state['market_mid']:.2f}** | Days to Expiry: **{days_to_exp}**")

# Prepare Tables
strikes_to_show = [target_k + (i * strike_step) for i in range(-5, 6)]
pricing_data = []
greeks_data = []

for k in strikes_to_show:
    # Price Sensitivity
    p_main = black_scholes_price(current_price, k, T, 0.043, final_iv, 0.0, opt_type)
    pricing_data.append({
        "Strike": f"${k:,.2f}",
        f"Price ({final_iv*100:.1f}% IV)": f"${p_main:.2f}"
    })
    
    # Greeks
    d, g, th = get_greeks(current_price, k, T, 0.043, final_iv, 0.0, opt_type)
    greeks_data.append({
        "Strike": f"${k:,.2f}",
        "Delta": f"{d:.4f}",
        "Gamma": f"{g:.4f}",
        "Theta (Daily)": f"${th:.4f}"
    })

t1, t2 = st.tabs(["💰 Pricing Matrix", "📈 Greeks Matrix"])
with t1:
    st.table(pd.DataFrame(pricing_data))
with t2:
    st.table(pd.DataFrame(greeks_data))
