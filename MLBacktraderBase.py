# make the necessary imports 
"""
!pip install backtrader
!pip install git+https://github.com/quantopian/pyfolio
!pip install yfinance
!pip install joblib
"""
from IPython.display import display
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

import pickle
import joblib

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

    def download_data(self, cols="Adj Close"):
        """
        yfinance에서 데이터 파일 다운로드하는 함수
        cols: 다운로드할 칼럼명 목록
        """

        self.data = yf.download(self.ticker, progress=True, actions=True, start=self.start, end=self.end)[cols]
        self.data = pd.DataFrame(self.data)
        self.data.rename(columns={"Adj Close":self.ticker}, inplace=True)

    def create_market_direction(self):
        """
        주가 변화율 기반으로 시장의 방향성 라벨링하는 함수
        """

        self.data["returns"] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace=True)
        self.data["direction"] = np.sign(self.data["returns"]).astype(int)


    def save_price_daily_returns_chart_image(self, filename="price_daily_returns.png", image_to_file=False):
        """
        일변 주가 변동 및 가격 변화율 시각화하는 함수
        data: 주가 데이터
        filename: 저장할 파일명
        image_to_file: 차트 이미지를 파일로 저장할지 여부, 기본값 False
        """

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

        if image_to_file:
            plt.savefig(filename, dpi=100)

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
        """
        주가 변화율 데이터를 bin 값으로 변환
        현재 설정은 0과 1로 변환(+변화율이면 1, -변화율이면 0으로 변환)
        """

        self.cols_bin = []
        for col in self.cols:
            col_bin = col + "_bin"
            self.data[col_bin] = np.digitize(self.data[col], bins=bins)
            self.cols_bin.append(col_bin)

    def training_models(self):
        """
        ML 모델학습을 시키는 함수
        """

        mfit = {model: self.models[model].fit(self.data[self.cols_bin], self.data["direction"]) for model in self.models.keys()}

    def models_prediction_position(self):
        """
        각 ML 모델별 예측값 생성
        """

        for model in self.models.keys():
            self.data["pos_" + model] = self.models[model].predict(self.data[self.cols_bin])

    def models_performance_evaluation(self):
        """
        모델별 성능 평가
        """

        self.strategy_rtn = []
        for model in self.models.keys():
            col = "strategy_" + model
            self.data[col] = self.data["pos_" + model] * self.data["returns"]
            self.strategy_rtn.append(col)

        self.strategy_rtn.insert(0, "returns")

    def models_calculate_total_return_std(self):
        """
        각 전략 모델별 총 수익률 및 표준 편차 계산
        """

        print("\nTotal Returns: \n")
        print(self.data[self.strategy_rtn].sum().apply(np.exp))
        print("\nAnnual Volitility:")
        print(self.data[self.strategy_rtn].std() * 252 ** 0.5)      

    def number_of_trades(self):
        """
        모델별 트레이딩 횟수
        """

        print("Number of trades SVM = ", (self.data["pos_svm"].diff()!=0).sum())
        print("Number of trades Ramdom Forest = ",(self.data["pos_random_forest"].diff()!=0).sum())

    def disp_performacne(self, filename="Machine Learning Classifiers Return Comparison.png", image_to_file=False):
        """
        거래 전략의 벡터화 백테스트 및 시간 경과에 따른 성능 시각화
        """

        ax = self.data[self.strategy_rtn].cumsum().apply(np.exp).plot(figsize=(12, 6), 
                                                        title = "Machine Learning Classifiers Return Comparison")
        ax.set_ylabel("Cumulative Returns")
        ax.grid(True)
        plt.tight_layout()

        if image_to_file:
            plt.savefig(filename, dpi=100)       


    def get_data(self):
        """
        훈련용 데이터를 반환하는 함수
        """

        return self.data

    def save_models(self,):
        """
        모델 저장
        """

        for model in self.models:
            joblib.dump(self.models[model], f"{model}.pkl")
            print(f"{model} 저장을 완료하였습니다.")

    def load_model(self, model):
        """
        모델 로딩
        """

        return joblib.load(f"{model}.pkl") 

    def create_predict_data(self, data, model):
        """
        모델 예측값 생성
        """
        data["pos"] = model.predict(data[self.cols_bin])        

if __name__ == "__main__":   
    m = MLBacktraderBase("AAPL", datetime.datetime(2020,1,1), datetime.datetime(2022,1,1))
    m.download_data()
    m.create_market_direction()
    m.save_price_daily_returns_chart_image(image_to_file=True)
    m.create_lags_data()
    m.create_bins_data()
    m.training_models()
    m.models_prediction_position()
    m.models_performance_evaluation()
    m.models_calculate_total_return_std()
    m.number_of_trades()
    m.disp_performacne(image_to_file=True)
    m.save_models()
    display(m.load_model("svm"))

