# -*- coding: utf-8 -*-

import pandas as pd
df = pd.read_csv("http://mlr.cs.umass.edu/ml/machine-learning-databases/iris/iris.data", header=None)
df.tail()

import matplotlib.pyplot as plt
import numpy as np
# 1-100行目の目的変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1, Iris-virginicaをiに変換
y = np.where(y == 'Iris-setosa', -1, 1)
# 1-100行目の1, 3列目の抽出
X = df.iloc[0:100, [0, 2]].values

# setosaのplot (赤の〇)
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# versinicaのplot (青の×)
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# インスタンス化
import Perceptron
ppn = Perceptron.Perceptron(eta = 0.1, n_iter = 10)
# トレーニングデータへのモデルの適合
ppn.fit(X, y)
# epochと誤分類差の関係
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker = "o")
plt.xlabel('Epoch')
plt.ylabel('Number of missclassifications')
plt.show()
