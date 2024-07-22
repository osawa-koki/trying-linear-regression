import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# サンプルデータの生成
# Xは独立変数、yは従属変数
X = np.arange(0, 100).reshape(-1, 1)
y = X + np.random.normal(0, 10, 100).reshape(-1, 1)

# 線形回帰モデルの作成
model = LinearRegression()

# モデルの学習
model.fit(X, y)

# 予測
y_pred = model.predict(X)

# 結果の表示
plt.scatter(X, y, color='blue', label='data points')
plt.plot(X, y_pred, color='red', label='regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.savefig('./public/line-regression.png')

# 回帰係数と切片の表示
print("regression coefficient (slope):", model.coef_)
print("intercept (intercept):", model.intercept_)
