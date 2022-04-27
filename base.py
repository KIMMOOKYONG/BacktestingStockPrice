# make the necessary imports 
"""
!pip install backtrader
!pip install git+https://github.com/quantopian/pyfolio
!pip install yfinance
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yfinance as yf
import warnings
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import datetime
import pyfolio as pf
import backtrader as bt
from backtrader.feeds import PandasData

# set the style and ignore warnings
plt.style.use("seaborn-colorblind")
plt.rcParams['figure.figsize'] = [15, 8]
plt.rcParams.update({'font.size': 12}) 

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

class Base():
    def __init__(self, ticker, start, end):

        self.ticker = ticker
        self.start = start
        self.end = end
        self.data = None
        self.cols_bin = []
        self.strategy_rtn = []

        self.models = {
            "log_reg": linear_model.LogisticRegression(),
            "gauss_nb": GaussianNB(),
            "svm": SVC(),
            "random_forest": RandomForestClassifier(max_depth=10, n_estimators=100),
            "MLP": MLPClassifier(),
        }

    def download_data(self, cols="Adj Close"):

        data = yf.download(self.ticker, progress=True, actions=True, start=self.start, end=self.end)[",".join(cols)]
        data = pd.DataFrame(data)
        data.rename(columns={"Adj Close":self.ticker}, inplace=True)

        data["returns"] = np.log(data / data.shift(1))
        data.dropna(inplace=True)
        data["direction"] = np.sign(data["returns"]).astype(int)

        return data

    def save_price_daily_returns_chart_image(self, data, filename="price_daily_returns.png"):

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12,6))
        ax[0].plot(self.data[self.ticker], label=f"{self.ticker} Adj Close")
        ax[0].set(title=f"{self.ticker} Closing Price", ylabel="Price")
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(self.data["returns"], label="Daily Returns")
        ax[1].set(title=f"{self.ticker} Daily Returns", ylabel="Returns")
        ax[1].grid(True)
        ax[1].legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300)

    def create_lags_data(self):

        lags = [1,2,3,4,5]
        cols = []
        for lag in lags:
            col = f"rtn_lag{lag}"
            self.data[col] = self.data["returns"].shift(lag)
            cols.append(col)
            self.data.dropna(inplace=True)

    def create_bins_data(self, bins=[0]):

        self.cols_bin = []
        for col in cols:
            col_bin = col + "_bin"
            self.data[col_bin] = np.digitize(self.data[col], bins=bins)
            self.cols_bin.append(col_bin)

    def fit_models(self):

        mfit = {model: models[model].fit(self.data[cols_bin], self.data["direction"]) for model in models.keys()}

    def derive_positions(self):

        for model in models.keys():
            self.data["pos_" + model] = models[model].predict(self.data[self.cols_bin])

    def evaluate_strats(self):

        self.strategy_rtn = []
        for model in models.keys():
            col = "strategy_" + model
            self.data[col] = self.data["pos_" + model] * self.data["returns"]
            self.strategy_rtn.append(col)

        self.strategy_rtn.insert(0, "returns")
