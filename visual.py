import streamlit as st
import numpy as np
from BlackScholes import BlackScholes  # Assuming BlackScholes class is imported here

# Set page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricing Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for user input
st.sidebar.header("Option Inputs")

# User inputs for option parameters
current_price = st.sidebar.number_input("Current Price of Underlying Asset", min_value=1.0, value=100.0, step=1.0)
strike_price = st.sidebar.number_input("Strike Price", min_value=1.0, value=100.0, step=1.0)
time_to_maturity = st.sidebar.number_input("Time to Maturity (in years)", min_value=0.01, value=1.0, step=0.01)
volatility = st.sidebar.number_input("Volatility (%)", min_value=0.0, value=20.0, step=0.1)
interest_rate = st.sidebar.number_input("Risk-Free Interest Rate (%)", min_value=0.0, value=5.0, step=0.1)

# Purchase prices for call and put options
purchase_price_call = st.sidebar.number_input("Purchase Price for Call Option", min_value=0.0, value=5.0, step=0.1)
purchase_price_put = st.sidebar.number_input("Purchase Price for Put Option", min_value=0.0, value=5.0, step=0.1)

# Initialize Black-Scholes model
bs = BlackScholes(
    time_to_maturity,
    strike_price,
    current_price,
    volatility / 100,  # Converting percentage to decimal
    interest_rate / 100,  # Converting percentage to decimal
    purchase_price_call,
    purchase_price_put
)

# Option prices and P&L
call_price = bs.call_price
put_price = bs.put_price
call_pnl = bs.call_pnl
put_pnl = bs.put_pnl

# Display results
st.title("Black-Scholes Option Pricing Model")
st.write(f"**Current Price of Asset**: ${current_price}")
st.write(f"**Strike Price**: ${strike_price}")
st.write(f"**Time to Maturity**: {time_to_maturity} years")
st.write(f"**Volatility**: {volatility}%")
st.write(f"**Interest Rate**: {interest_rate}%")

# Option Prices
st.subheader("Option Prices")
st.write(f"**Call Option Price**: ${call_price:.2f}")
st.write(f"**Put Option Price**: ${put_price:.2f}")

# P&L Calculation
st.subheader("Profit and Loss (P&L) for Your Options")
st.write(f"**Call Option P&L**: ${call_pnl:.2f}")
st.write(f"**Put Option P&L**: ${put_pnl:.2f}")

# Visualizing the P&L as a heatmap (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Create a grid of strikes and underlying prices for the heatmap
strike_prices = np.linspace(50, 150, 100)
underlying_prices = np.linspace(50, 150, 100)
call_pnl_matrix = np.zeros((len(underlying_prices), len(strike_prices)))
put_pnl_matrix = np.zeros((len(underlying_prices), len(strike_prices)))

# Calculate P&L for call and put options across the grid
for i, S in enumerate(underlying_prices):
    for j, K in enumerate(strike_prices):
        bs_temp = BlackScholes(
            time_to_maturity,
            K,
            S,
            volatility / 100,
            interest_rate / 100,
            purchase_price_call,
            purchase_price_put
        )
        call_pnl_matrix[i, j] = bs_temp.call_pnl
        put_pnl_matrix[i, j] = bs_temp.put_pnl

# Plot Heatmap for Call Option P&L
plt.figure(figsize=(12, 6))
sns.heatmap(call_pnl_matrix, xticklabels=[f"{K:.2f}" for K in strike_prices], yticklabels=[f"{S:.2f}" for S in underlying_prices], cmap="RdYlGn", annot=True, fmt=".2f")
plt.title("Call Option P&L Heatmap")
plt.xlabel("Strike Price")
plt.ylabel("Underlying Asset Price")
st.pyplot()

# Plot Heatmap for Put Option P&L
plt.figure(figsize=(12, 6))
sns.heatmap(put_pnl_matrix, xticklabels=[f"{K:.2f}" for K in strike_prices], yticklabels=[f"{S:.2f}" for S in underlying_prices], cmap="RdYlGn", annot=True, fmt=".2f")
plt.title("Put Option P&L Heatmap")
plt.xlabel("Strike Price")
plt.ylabel("Underlying Asset Price")
st.pyplot()

# Footer with LinkedIn link
st.sidebar.markdown(
    """
    <hr>
    <center>
        Created by [Your Name](https://www.linkedin.com/in/your-linkedin-profile)
    </center>
    """, unsafe_allow_html=True
)
