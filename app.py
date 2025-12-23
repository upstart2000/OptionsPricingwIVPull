import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
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
        return brentq(objective, 1e-5, 5.0)
    except:
        return 0.40 # Fallback if math fails

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

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Auto-Greeks Solver", layout="wide")
st.title("🤖 Automated IV Solver & Greeks Chain")

with st.sidebar:
    st.header("1. Settings")
    ticker_sym = st.text_input("Ticker", value="AAPL").upper()
    ticker_obj = yf.Ticker(ticker_sym)
    
    try:
        current_price = ticker_obj.fast_info['lastPrice']
        st.success(f"Spot: ${current_price:.2f}")
    except:
        current_price = 150.0

    opt_mode = st.radio("Type", ["Call", "Put"])
    target_k = st.number_input("Strike ($)", value=float(np.round(current_price, 0)))
    
    available_exps = ticker_obj.options
    selected_exp = st.selectbox("Expiration", available_exps)
    expiry_dt = date.fromisoformat(selected_exp)
    T = max((expiry_dt - date.today()).days, 0) / 365.0

    # Auto-Fetch Logic
    if st.button("⚡ Fetch & Solve Live IV"):
        try:
            chain = ticker_obj.option_chain(selected_exp)
            df = chain.calls if opt_mode == "Call" else chain.puts
            # Find exact or closest strike
            row = df.iloc[(df['strike'] - target_k).abs().idxmin()]
            
            mid = (row['bid'] + row['ask']) / 2
            st.session_state['market_mid'] = mid
            st.session_state['market_bid'] = row['bid']
            st.session_state['market_ask'] = row['ask']
            
            # Solve for IV based on Midpoint
            r_val, q_val = 0.042, 0.005 # Approximations
            solved_iv = find_implied_vol(mid, current_price, target_k, T, r_val, q_val, opt_mode)
            st.session_state['solved_iv'] = solved_iv
            st.success(f"IV Solved from ${mid:.2f} Midpoint")
        except:
            st.error("Could not fetch market midpoint.")

    # Values used for tables
    final_iv = st.number_input("Implied Volatility", value=st.session_state.get('solved_iv', 0.40), format="%.4f")
    r_rate = st.number_input("Risk-Free Rate", value=0.043, format="%.3f")
    div_yield = st.number_input("Div Yield", value=0.005, format="%.3f")
    strike_step = st.number_input("Step", value=5.0)

# --- 3. RESULTS ---
if 'market_mid' in st.session_state:
    st.write(f"**Market Context:** Bid: `${st.session_state['market_bid']}` | Ask: `${st.session_state['market_ask']}` | Mid: **`${st.session_state['market_mid']:.2f}`**")

strikes = [target_k + (i * strike_step) for i in range(-5, 6)]
p_data, g_data = [], []

for k in strikes:
    p_base = black_scholes_price(current_price, k, T, r_rate, final_iv, div_yield, opt_mode)
    p_data.append({"Strike": f"${k:,.2f}", f"{final_iv*100-1:.1f}% IV": f"${black_scholes_price(current_price, k, T, r_rate, final_iv-0.01, div_yield, opt_mode):.2f}", f"{final_iv*100:.1f}% IV": f"${p_base:.2f}", f"{final_iv*100+1:.1f}% IV": f"${black_scholes_price(current_price, k, T, r_rate, final_iv+0.01, div_yield, opt_mode):.2f}"})
    
    d, g, th = get_greeks(current_price, k, T, r_rate, final_iv, div_yield, opt_mode)
    g_data.append({"Strike": f"${k:,.2f}", "Delta": f"{d:.4f}", "Gamma": f"{g:.4f}", "Theta (Daily)": f"${th:.4f}"})

t1, t2 = st.tabs(["💰 Pricing Matrix", "📈 Greeks Matrix"])
with t1: st.table(pd.DataFrame(p_data))
with t2: st.table(pd.DataFrame(g_data))
