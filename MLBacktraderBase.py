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

class MLBacktraderBase():
    def __init__(self, ticker, start, end):

        self.ticker = ticker # 종목코드
        self.start = start # 시작일
        self.end = end # 종료일
        self.data = None # 데이터
        self.cols = [] # 주가 변화율 계산 칼럼명
        self.cols_bin = [] # 주가 변화율 계산 바이너리(0,1) 칼럼명
        self.strategy_rtn = []

        self.models = {
            "log_reg": linear_model.LogisticRegression(),
            "gauss_nb": GaussianNB(),
            "svm": SVC(),
            "random_forest": RandomForestClassifier(max_depth=10, n_estimators=100),
            "MLP": MLPClassifier(),
        }

    def download_data(self, cols=["Adj Close"]):
        """
        yfinance에서 데이터 파일 다운로드하는 함수
        cols: 다운로드할 칼럼명 목록
        """

        self.data = yf.download(self.ticker, progress=True, actions=True, start=self.start, end=self.end)[",".join(cols)]
        self.data = pd.DataFrame(data)
        self.data.rename(columns={"Adj Close":self.ticker}, inplace=True)

        self.data["returns"] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace=True)
        self.data["direction"] = np.sign(self.data["returns"]).astype(int)

    def save_price_daily_returns_chart_image(self, data, filename="price_daily_returns.png", image_to_file=False):
        """
        일변 주가 변동 및 가격 변화율 시각화하는 함수
        data: 주가 데이터
        filename: 저장할 파일명
        image_to_file: 차트 이미지를 파일로 저장할지 여부, 기본값 False
        """

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12,6))
        ax[0].plot(data[self.ticker], label=f"{self.ticker} Adj Close")
        ax[0].set(title=f"{self.ticker} Closing Price", ylabel="Price")
        ax[0].grid(True)
        ax[0].legend()

        ax[1].plot(data["returns"], label="Daily Returns")
        ax[1].set(title=f"{self.ticker} Daily Returns", ylabel="Returns")
        ax[1].grid(True)
        ax[1].legend()

        plt.tight_layout()

        if image_to_file:
            plt.savefig(filename, dpi=300)

    def create_lags_data(self, lags=[1,2,3,4,5]):
        """
        기간별 주가 변화율을 계산하는 함수
        lags: 주가 변화율을 계산할 기간 설정
        """

        lags = lags
        self.cols = []
        for lag in lags:
            col = f"rtn_lag{lag}"
            self.data[col] = self.data["returns"].shift(lag)
            self.cols.append(col)
            self.data.dropna(inplace=True)

    def create_bins_data(self, bins=[0]):

        self.cols_bin = []
        for col in self.cols:
            col_bin = col + "_bin"
            self.data[col_bin] = np.digitize(self.data[col], bins=bins)
            self.cols_bin.append(col_bin)

    def fit_models(self):

        mfit = {model: self.models[model].fit(self.data[self.cols_bin], self.data["direction"]) for model in self.models.keys()}

    def derive_positions(self):

        for model in self.models.keys():
            self.data["pos_" + model] = self.models[model].predict(self.data[self.cols_bin])

    def evaluate_strats(self):

        self.strategy_rtn = []
        for model in self.models.keys():
            col = "strategy_" + model
            self.data[col] = self.data["pos_" + model] * self.data["returns"]
            self.strategy_rtn.append(col)

        self.strategy_rtn.insert(0, "returns")