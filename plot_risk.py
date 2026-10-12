"""
plot_risk.py - publication-style rolling volatility and correlation charts.

Expects `prices`: a DataFrame of daily closes with a DatetimeIndex and one
column per ticker (exactly what yf.download(...)["Close"] gives you).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

TRADING_DAYS = 252
SOURCE = "Source: Yahoo Finance via yfinance"


# ---------------------------------------------------------------- style
def set_style():
    """One house style for every chart, instead of matplotlib defaults."""
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",      # swap for "Inter"/"Helvetica" if installed
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10,
        "axes.labelcolor": "#444444",
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
    })


def _title(ax, title, subtitle):
    ax.set_title(title, pad=30)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes,
            fontsize=10, color="#666666", va="bottom")


def _footnote(fig, text):
    fig.text(0.01, -0.02, text, fontsize=8, color="#888888", ha="left")


def log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna(how="all")


# ------------------------------------------------------ rolling volatility
def plot_rolling_vol(prices, tickers, short=21, long=63, save_as=None):
    """
    Annualised rolling volatility for a few highlighted stocks:
      - thin faint line  = short window (~1 month), shows spikes
      - thick line       = long window (~3 months), shows the regime
      - grey band        = interquartile range of every stock in `prices`,
                           so you can see if a name is unusually risky
    """
    rets = log_returns(prices)
    ann = np.sqrt(TRADING_DAYS)

    all_long = rets.rolling(long, min_periods=long).std() * ann
    q25, q75 = all_long.quantile(0.25, axis=1), all_long.quantile(0.75, axis=1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(q25.index, q25, q75, color="#d9d9d9", alpha=0.6, lw=0,
                    label=f"Middle 50% of {prices.shape[1]} stocks")

    palette = sns.color_palette("deep", n_colors=len(tickers))
    for colour, t in zip(palette, tickers):
        vol_s = rets[t].rolling(short, min_periods=short).std() * ann
        vol_l = all_long[t]
        ax.plot(vol_s.index, vol_s, color=colour, lw=0.8, alpha=0.35)
        ax.plot(vol_l.index, vol_l, color=colour, lw=2.2)
        last = vol_l.dropna()
        # label the line directly instead of using a legend box
        ax.annotate(t.replace(".L", ""), xy=(last.index[-1], last.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=colour, fontsize=9, fontweight="bold")

    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(bottom=0)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.margins(x=0.01)
    ax.set_ylabel("Annualised volatility")
    ax.legend(loc="upper left")

    _title(ax, "Rolling volatility",
           f"{long}-day (bold) and {short}-day (faint) annualised, "
           f"{rets.index[0]:%b %Y} – {rets.index[-1]:%b %Y}")
    _footnote(fig, f"{SOURCE}. Annualised = std of daily log returns × √{TRADING_DAYS}.")

    if save_as:
        fig.savefig(save_as)
    return fig, ax


# ------------------------------------------------------ correlation matrix
def plot_corr(prices, method="pearson", save_as=None):
    """
    Correlation of daily log returns (never prices - prices trend together
    and give fake high correlations).
      - clustered ordering so similar stocks sit next to each other
      - lower triangle only, diagonal dropped (it's always 1)
      - diverging colours fixed at -1..1 and centred on 0, so the colour
        scale means the same thing in every chart you make
    """
    rets = log_returns(prices)
    corr = rets.corr(method=method, min_periods=TRADING_DAYS // 2).clip(-1, 1)

    dist = np.sqrt(0.5 * (1 - corr))
    order = leaves_list(linkage(squareform(dist.values, checks=False), "average"))
    corr = corr.iloc[order, order]
    corr.index = corr.columns = [c.replace(".L", "") for c in corr.columns]

    tri = corr.iloc[1:, :-1]                         # drop empty row/col
    mask = np.triu(np.ones_like(tri, dtype=bool), k=1)

    n = len(tri)
    side = max(6, 0.55 * n)
    fig, ax = plt.subplots(figsize=(side + 1.5, side))
    cmap = sns.diverging_palette(240, 15, s=85, l=45, as_cmap=True)
    sns.heatmap(tri, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.8, linecolor="white",
                annot=n <= 15, fmt=".2f", annot_kws={"size": 8},
                cbar_kws={"shrink": 0.6, "label": "Correlation"}, ax=ax)
    ax.grid(False)
    ax.tick_params(length=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)

    _title(ax, "Return correlations",
           f"{method.title()} on daily log returns, "
           f"{rets.index[0]:%b %Y} – {rets.index[-1]:%b %Y}, clustered")
    _footnote(fig, SOURCE + ".")

    if save_as:
        fig.savefig(save_as)
    return fig, ax


# ---------------------------------------------------------------- example
if __name__ == "__main__":
    import yfinance as yf

    tickers = ["HSBA.L", "BARC.L", "LLOY.L", "SHEL.L", "BP.L",
               "AZN.L", "GSK.L", "ULVR.L", "RIO.L", "GLEN.L"]
    prices = yf.download(tickers, start="2021-01-01",
                         auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna(how="all")

    set_style()
    plot_rolling_vol(prices, ["BARC.L", "SHEL.L", "AZN.L"], save_as="rolling_vol.png")
    plot_corr(prices, save_as="corr_matrix.png")
    plt.show()
