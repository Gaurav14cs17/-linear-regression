import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('H:\p2.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values



#Linear Reg
from sklearn.linear_model import LinearRegression
linear_Reg = LinearRegression()
linear_Reg.fit(X , Y)
pred_Y = linear_Reg.predict(X)


plt.scatter(X , Y , color= 'Red')
plt.plot(X , pred_Y)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(4)

X_poly = poly.fit_transform(X)
poly.fit( X_poly , Y)


p_lin_reg = LinearRegression()
p_lin_reg.fit(X_poly , Y)
pol_pred_y = p_lin_reg.predict(X_poly)

plt.scatter(X , Y , color= 'Red')
plt.plot(X , pred_Y)
plt.plot(X, pol_pred_y, color = 'Black')


linear_Reg.predict(6.8)
p_lin_reg.predict(poly.fit_transform(6.8))


