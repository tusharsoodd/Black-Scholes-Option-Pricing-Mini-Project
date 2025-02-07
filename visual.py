import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from blackScholes import BlackScholes
#######################
# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: auto;
    margin: 0 auto;
}
.metric-call { background-color: #90ee90; color: black; margin-right: 10px; border-radius: 10px; }
.metric-put { background-color: #ffcccb; color: black; border-radius: 10px; }
.metric-value { font-size: 1.5rem; font-weight: bold; margin: 0; }
.metric-label { font-size: 1rem; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# LinkedIn link
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-top: 20px;">
        <strong>Created by <a href="https://www.linkedin.com/in/tusharsoodd/" target="_blank">Tushar Sood</a></strong>
    </div>
    """,
    unsafe_allow_html=True,
)



# Sidebar Inputs
with st.sidebar:
    st.title("Black-Scholes Model")

    current_price = st.number_input("Current Asset Price", value=100.0)
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)

    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

# Compute Prices
bs_model = BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate)
call_price = bs_model.call_price
put_price = bs_model.put_price

# Display Prices
col1, col2 = st.columns([1,1])
col1.markdown(f"""<div class="metric-container metric-call"><div><div class="metric-label">CALL Value</div><div class="metric-value">${call_price:.2f}</div></div></div>""", unsafe_allow_html=True)
col2.markdown(f"""<div class="metric-container metric-put"><div><div class="metric-label">PUT Value</div><div class="metric-value">${put_price:.2f}</div></div></div>""", unsafe_allow_html=True)

# Heatmap Function
def compute_option_prices_vectorized(spot_range, vol_range, time_to_maturity, strike, interest_rate):
    spot_grid, vol_grid = np.meshgrid(spot_range, vol_range)
    d1 = (np.log(spot_grid / strike) + (interest_rate + 0.5 * vol_grid**2) * time_to_maturity) / (vol_grid * np.sqrt(time_to_maturity))
    d2 = d1 - vol_grid * np.sqrt(time_to_maturity)

    call_prices = spot_grid * norm.cdf(d1) - strike * np.exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
    put_prices = strike * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2) - spot_grid * norm.cdf(-d1)

    return call_prices, put_prices

# Heatmap Plot Function
def plot_heatmap(data, spot_range, vol_range, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data, 
                xticklabels=np.round(spot_range, 2), 
                yticklabels=np.round(vol_range, 2), 
                annot=True, 
                fmt=".2f", 
                cmap="RdYlGn",  # Red for low, Yellow for mid, Green for high
                ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Volatility")
    return fig


# Compute Heatmap Data
call_prices, put_prices = compute_option_prices_vectorized(spot_range, vol_range, time_to_maturity, strike, interest_rate)

# Display Heatmaps
st.title("Options Price - Interactive Heatmap")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels.")

col1, col2 = st.columns([1,1])
col1.subheader("Call Price Heatmap")
col1.pyplot(plot_heatmap(call_prices, spot_range, vol_range, "CALL"))

col2.subheader("Put Price Heatmap")
col2.pyplot(plot_heatmap(put_prices, spot_range, vol_range, "PUT"))
