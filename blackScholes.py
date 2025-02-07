import numpy as np
from scipy.stats import norm


class BlackScholes:
    def __init__(
        self,
        time_to_maturity: float,
        strike: float,
        current_price: float,
        volatility: float,
        interest_rate: float,
        purchase_price_call: float = 0.0,  # Purchase price for call
        purchase_price_put: float = 0.0,   # Purchase price for put
    ):
        self.T = time_to_maturity
        self.K = strike
        self.S = current_price
        self.sigma = volatility
        self.r = interest_rate
        self.purchase_price_call = purchase_price_call
        self.purchase_price_put = purchase_price_put

    @property
    def d1(self):
        """Calculate d1 for Black-Scholes formula."""
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * np.sqrt(self.T)
        )

    @property
    def d2(self):
        """Calculate d2 for Black-Scholes formula."""
        return self.d1 - self.sigma * np.sqrt(self.T)

    @property
    def call_price(self):
        """Calculate European call option price."""
        return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)

    @property
    def put_price(self):
        """Calculate European put option price."""
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

    @property
    def call_delta(self):
        """Calculate Delta for call option."""
        return norm.cdf(self.d1)

    @property
    def put_delta(self):
        """Calculate Delta for put option."""
        return norm.cdf(self.d1) - 1

    @property
    def gamma(self):
        """Calculate Gamma (same for calls and puts)."""
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    @property
    def vega(self):
        """Calculate Vega (sensitivity to volatility)."""
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T)

    @property
    def call_theta(self):
        """Calculate Theta for call option."""
        first_term = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        second_term = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        return first_term - second_term

    @property
    def put_theta(self):
        """Calculate Theta for put option."""
        first_term = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        second_term = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return first_term + second_term

    @property
    def call_rho(self):
        """Calculate Rho for call option (sensitivity to interest rates)."""
        return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)

    @property
    def put_rho(self):
        """Calculate Rho for put option."""
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)

    @property
    def call_pnl(self):
        """Calculate P&L for call option."""
        return self.call_price - self.purchase_price_call

    @property
    def put_pnl(self):
        """Calculate P&L for put option."""
        return self.put_price - self.purchase_price_put
