import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import yfinance as yf

stocks = input("Enter The Code Of Stock")
data = yf.download(stocks, auto_adjust=True)
data.tail()

data.Close.plot(figsize=(10,7), color = 'r')
plt.ylabel(" {} Prices".format(stocks))
plt.show()

x = data.drop(["Close"], axis =1)
y = data["Close"]

from sklearn.model_selection import train_test_split

X_train,  X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)

X_test.fillna(X_train.mean(), inplace=True)
pred = lr.predict(X_test)
print(pred[0])