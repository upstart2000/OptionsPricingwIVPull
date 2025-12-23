# 📈 Automated Options IV Solver & Greeks Lab

A professional-grade financial tool built with **Python** and **Streamlit** that reverse-engineers market data to find the "True" Implied Volatility (IV) using the **Black-Scholes-Merton model**. 

Unlike standard tools, this app fetches live Bid/Ask midpoints and uses numerical root-finding (Brent's Method) to provide institutional-grade accuracy for Option Greeks and pricing sensitivity.

## 🚀 Key Features
- **Automated IV Discovery:** Fetches live Bid/Ask spreads from Yahoo Finance and solves for IV based on the midpoint price.
- **Risk Management (Greeks):** Real-time calculation of **Delta**, **Gamma**, and **Theta** (daily decay).
- **Volatility Sensitivity:** Generates a pricing matrix showing how a +/- 1% shift in IV affects premiums across 11 strikes.
- **Dynamic Refresh:** Includes a dedicated refresh button to update stock spot prices instantly.
- **Precision Inputs:** Supports 3-decimal precision for Risk-Free Rates and Dividend Yields.

## 🛠️ Installation & Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
