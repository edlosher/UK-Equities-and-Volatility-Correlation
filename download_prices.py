import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from plot_risk import set_style, plot_rolling_vol, plot_corr

tickers = [
    "AZN.L", "GSK.L",            
    "SHEL.L", "BP.L",            
    "HSBA.L", "BARC.L", "LLOY.L",
    "LSEG.L",                     
    "RIO.L", "GLEN.L",           
    "ULVR.L", "DGE.L", "TSCO.L", 
    "RR.L", "NG.L",               
]

raw = yf.download(tickers, start="2021-01-01", auto_adjust=True, progress=False)
prices = raw["Close"]

failed = prices.columns[prices.isna().all()].tolist()
print("Failed:", failed or "none")
print(prices.tail().round(2))

prices.to_csv("ftse_prices.csv")
returns = prices.pct_change(fill_method=None).dropna(how="all")
print(returns.describe().T[["mean", "std"]])

set_style()

cumulative = (1 + returns).cumprod()
cumulative.plot(figsize=(12, 6), title="Growth of £1")

rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
rolling_vol.plot(figsize=(12, 6), title="30-day rolling volatility")

plot_rolling_vol(prices, ["BARC.L", "SHEL.L", "AZN.L"], save_as="rolling_vol.png")
plot_corr(prices, save_as="corr_matrix.png")

# Risk/return summary
ann_return = returns.mean() * 252
ann_vol = returns.std() * np.sqrt(252)
sharpe = ann_return / ann_vol
max_dd = (cumulative / cumulative.cummax() - 1).min()

summary = pd.DataFrame({
    "Ann. return": ann_return,
    "Ann. vol": ann_vol,
    "Sharpe": sharpe,
    "Max drawdown": max_dd,
}).sort_values("Sharpe", ascending=False)

print(summary.round(2))

plt.show()
