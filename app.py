import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import date

# --- 1. CORE MATH FUNCTIONS (Black-Scholes + Greeks) ---
def get_greeks(S, K, T, r, sigma, q, opt_type="Call"):
    T_val = max(T, 1e-10)
    sigma_val = max(sigma, 1e-4)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_val**2) * T_val) / (sigma_val * np.sqrt(T_val))
    d2 = d1 - sigma_val * np.sqrt(T_val)
    
    # Delta
    if opt_type == "Call":
        delta = np.exp(-q * T_val) * norm.cdf(d1)
    else:
        delta = np.exp(-q * T_val) * (norm.cdf(d1) - 1)
        
    # Gamma (Same for Call and Put)
    gamma = (norm.pdf(d1) * np.exp(-q * T_val)) / (S * sigma_val * np.sqrt(T_val))
    
    # Theta (Annualized, then divided by 365 for daily)
    term1 = -(S * sigma_val * np.exp(-q * T_val) * norm.pdf(d1)) / (2 * np.sqrt(T_val))
    if opt_type == "Call":
        term2 = q * S * np.exp(-q * T_val) * norm.cdf(d1)
        term3 = r * K * np.exp(-r * T_val) * norm.cdf(d2)
        theta = (term1 + term2 - term3) / 365.0
    else:
        term2 = q * S * np.exp(-q * T_val) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T_val) * norm.cdf(-d2)
        theta = (term1 - term2 + term3) / 365.0
        
    return delta, gamma, theta

def black_scholes_price(S, K, T, r, sigma, q, opt_type="Call"):
    T_val = max(T, 1e-10)
    sigma_val = max(sigma, 1e-4)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_val**2) * T_val) / (sigma_val * np.sqrt(T_val))
    d2 = d1 - sigma_val * np.sqrt(T_val)
    if opt_type == "Call":
        return (S * np.exp(-q * T_val) * norm.cdf(d1)) - (K * np.exp(-r * T_val) * norm.cdf(d2))
    else:
        return (K * np.exp(-r * T_val) * norm.cdf(-d2)) - (S * np.exp(-q * T_val) * norm.cdf(-d1))

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Options Master Tool", layout="wide")
st.title("🛡️ Professional Options Pricing & Greeks Matrix")

with st.sidebar:
    st.header("1. Market Data")
    col_tk, col_ref = st.columns([3, 1])
    with col_tk:
        ticker_sym = st.text_input("Stock Ticker", value="NVDA").upper()
    with col_ref:
        st.write(" ") 
        st.button("🔄")

    ticker_obj = yf.Ticker(ticker_sym)
    try:
        current_price = ticker_obj.fast_info['lastPrice']
        st.success(f"Current {ticker_sym} Price: ${current_price:.2f}")
    except:
        current_price = 100.0
        st.error("Ticker error. Using default.")

    st.header("2. Option Config")
    opt_mode = st.radio("Option Type", ["Call", "Put"])
    target_strike = st.number_input("Central Strike ($)", value=float(np.round(current_price, 0)))
    strike_step = st.number_input("Strike Variation ($)", value=5.0)
    
    available_expiries = ticker_obj.options
    selected_exp = st.selectbox("Select Expiration", available_expiries)

    if 'iv_input' not in st.session_state:
        st.session_state['iv_input'] = 40.0
        
    if st.button("🔍 Fetch Market IV"):
        try:
            chain = ticker_obj.option_chain(selected_exp)
            df_chain = chain.calls if opt_mode == "Call" else chain.puts
            idx = (df_chain['strike'] - target_strike).abs().idxmin()
            st.session_state['iv_input'] = df_chain.loc[idx, 'impliedVolatility'] * 100
            st.success(f"Market IV Found!")
        except:
            st.error("Market IV lookup failed.")

    iv_final = st.number_input("Implied Volatility %", value=st.session_state['iv_input'], step=0.1)
    r_rate = st.number_input("Risk-Free Rate", value=0.043, format="%.3f")
    div_yield = st.number_input("Dividend Yield", value=0.000, format="%.3f")

# --- 3. CALCULATIONS ---
expiry_dt = date.fromisoformat(selected_exp)
dte = max((expiry_dt - date.today()).days, 0)
T = dte / 365.0
vol = iv_final / 100.0
strikes = [target_strike + (i * strike_step) for i in range(-5, 6)]

# --- 4. GENERATE TABLES ---
price_rows, greeks_rows = [], []

for k in strikes:
    # Price Sensitivity Table
    p_base = black_scholes_price(current_price, k, T, r_rate, vol, div_yield, opt_mode)
    p_low = black_scholes_price(current_price, k, T, r_rate, vol - 0.01, div_yield, opt_mode)
    p_high = black_scholes_price(current_price, k, T, r_rate, vol + 0.01, div_yield, opt_mode)
    
    price_rows.append({
        "Strike": f"${k:,.2f}",
        f"{iv_final-1:.1f}% IV": f"${p_low:.2f}",
        f"{iv_final:.1f}% IV": f"${p_base:.2f}",
        f"{iv_final+1:.1f}% IV": f"${p_high:.2f}"
    })
    
    # Greeks Table
    d, g, th = get_greeks(current_price, k, T, r_rate, vol, div_yield, opt_mode)
    greeks_rows.append({
        "Strike": f"${k:,.2f}",
        "Chosen IV": f"{iv_final:.1f}%",
        "Delta": f"{d:.4f}",
        "Gamma": f"{g:.4f}",
        "Theta (Daily)": f"${th:.4f}"
    })

# --- 5. DISPLAY ---
tab1, tab2 = st.tabs(["💰 Pricing Matrix", "📈 Option Greeks"])

with tab1:
    st.subheader(f"Price Sensitivity Table (+/- 1% IV)")
    st.table(pd.DataFrame(price_rows))

with tab2:
    st.subheader(f"Risk Metrics (Greeks) at {iv_final:.1f}% IV")
    st.table(pd.DataFrame(greeks_rows))

st.divider()
st.caption(f"Note: Delta and Gamma are calculated per share. Theta is the dollar decay per share per day.")
