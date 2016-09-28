import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# Read Data
data = pd.read_csv("ex1data1.txt", header=None)

# draw scatter
plt.scatter(data[0], data[1], marker="x", c="r")
plt.xlabel("Population of city in 10,000s")
plt.ylabel("Profit in $10,000s")

X = np.array([data[0]]).T
y = np.array(data[1])

# Model Fitting by sklearn
model = linear_model.LinearRegression()
model.fit(X, y)

px = np.arange(X.min(),X.max(),.01)[:,np.newaxis]
py = model.predict(px)

# draw line
plt.plot(px, py, color="blue", linewidth=3)
plt.show()
