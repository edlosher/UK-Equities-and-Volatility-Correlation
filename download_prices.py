import yfinance as yf

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
returns = prices.pct_change().dropna()

print(returns.describe().T[["mean", "std"]])

# Growth of £1 invested at the start
cumulative = (1 + returns).cumprod()
cumulative.plot(figsize=(12, 6), title="Growth of £1")

import matplotlib.pyplot as plt
plt.show()