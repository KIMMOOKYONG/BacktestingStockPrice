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

class SignalData(PandasData):
    """
    백테스팅 플랫폼에 데이터를 공급하는 클래스
    데이터프레임 구조 정의
    """

    ohlcv = ["open","high","low","close","volume"]
    cols = ohlcv + ["predicted"]
    
    # create lines
    lines = tuple(cols)

    # define parameters
    params = {c:-1 for c in cols}
    params.update({"datetime":None})
    params = tuple(params.items())
