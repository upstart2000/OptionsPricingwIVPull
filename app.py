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
        return brentq(objective, 1e-5, 5.0)
    except:
        return 0.40 # Fallback

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
st.set_page_config(page_title="Professional Options Solver", layout="wide")
st.title("🛡️ Automated Options IV Solver & Greeks Matrix")

with st.sidebar:
    st.header("1. Market Data")
    col_tk, col_ref = st.columns([3, 1])
    with col_tk:
        ticker_sym = st.text_input("Stock Ticker", value="AAPL").upper()
    with col_ref:
        st.write(" ") 
        st.button("🔄") # Dedicated Spot Price Refresh

    ticker_obj = yf.Ticker(ticker_sym)
    try:
        current_price = ticker_obj.fast_info['lastPrice']
        st.success(f"Spot Price: ${current_price:.2f}")
    except:
        current_price = 150.0
        st.error("Ticker fetch error.")

    st.header("2. Option Configuration")
    opt_mode = st.radio("Option Type", ["Call", "Put"])
    target_k = st.number_input("Central Strike ($)", value=float(np.round(current_price, 0)))
    
    available_exps = ticker_obj.options
    selected_exp = st.selectbox("Select Expiration", available_exps)
    expiry_dt = date.fromisoformat(selected_exp)
    T = max((expiry_dt - date.today()).days, 0) / 365.0

    st.header("3. IV Solver")
    if st.button("⚡ Solve IV from Market Midpoint"):
        try:
            chain = ticker_obj.option_chain(selected_exp)
            df = chain.calls if opt_mode == "Call" else chain.puts
            row = df.iloc[(df['strike'] - target_k).abs().idxmin()]
            
            mid = (row['bid'] + row['ask']) / 2
            st.session_state['market_mid'] = mid
            st.session_state['market_bid'] = row['bid']
            st.session_state['market_ask'] = row['ask']
            
            # Solve for IV
            r_temp, q_temp = 0.043, 0.005
            solved_iv = find_implied_vol(mid, current_price, target_k, T, r_temp, q_temp, opt_mode)
            st.session_state['solved_iv'] = solved_iv
            st.success("IV Successfully Solved")
        except:
            st.error("Market IV Fetch Failed.")

    # Main IV Input (Updates from solver but allows manual override)
    if 'solved_iv' not in st.session_state:
        st.session_state['solved_iv'] = 0.40
    
    final_iv = st.number_input("Implied Volatility (Manual or Solved)", value=float(st.session_state['solved_iv']), format="%.4f")
    
    st.header("4. Parameters")
    r_rate = st.number_input("Risk-Free Rate", value=0.043, format="%.3f")
    div_yield = st.number_input("Dividend Yield", value=0.005, format="%.3f")
    strike_step = st.number_input("Strike Step ($)", value=5.0)

# --- 3. DATA GENERATION ---
if 'market_mid' in st.session_state:
    st.info(f"Market Context for ${target_k} Strike: Bid ${st.session_state['market_bid']} | Ask ${st.session_state['market_ask']} | Midpoint **${st.session_state['market_mid']:.2f}**")

strikes = [target_k + (i * strike_step) for i in range(-5, 6)]
p_rows, g_rows = [], []

for k in strikes:
    # Pricing Matrix Calculation
    p_base = black_scholes_price(current_price, k, T, r_rate, final_iv, div_yield, opt_mode)
    p_low = black_scholes_price(current_price, k, T, r_rate, final_iv - 0.01, div_yield, opt_mode)
    p_high = black_scholes_price(current_price, k, T, r_rate, final_iv + 0.01, div_yield, opt_mode)
    
    p_rows.append({
        "Strike": f"${k:,.2f}",
        f"{final_iv*100-1:.1f}% IV": f"${p_low:.2f}",
        f"{final_iv*100:.1f}% IV": f"${p_base:.2f}",
        f"{final_iv*100+1:.1f}% IV": f"${p_high:.2f}"
    })
    
    # Greeks Matrix Calculation
    d, g, th = get_greeks(current_price, k, T, r_rate, final_iv, div_yield, opt_mode)
    g_rows.append({
        "Strike": f"${k:,.2f}",
        "Delta": f"{d:.4f}",
        "Gamma": f"{g:.4f}",
        "Theta (Daily)": f"${th:.4f}"
    })

# --- 4. OUTPUT DISPLAY ---
tab1, tab2 = st.tabs(["💰 Pricing Sensitivity Matrix", "📈 Risk Metrics (Greeks)"])

with tab1:
    st.subheader("Price Sensitivity (+/- 1% Volatility Shift)")
    st.table(pd.DataFrame(p_rows))

with tab2:
    st.subheader(f"Greeks Matrix at {final_iv*100:.2f}% IV")
    st.table(pd.DataFrame(g_rows))

st.divider()
st.caption("Pricing Model: Black-Scholes-Merton | IV Solver:
