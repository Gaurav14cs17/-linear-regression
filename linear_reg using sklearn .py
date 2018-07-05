import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def solve(X,Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    
    x =y = 0
    n = len(X)
    for i in range(n):
        x = x + (X[i] - x_mean) * (Y[i]-y_mean)
        y = y +(X[i] - x_mean)**2
        
    m = x/y
    c = (y_mean - m*x_mean)
    
    min_x = np.min(X)
    max_x = np.max(X)
    
    x = np.linspace(min_x, max_x, 1000)
    y = c + m*x
    plt.plot(x,y,c='Red')
    plt.scatter(X,Y , c='Blue')
    plt.legend()
    plt.show()
    
    
    
    
dataset = pd.read_csv('H:\ML PDF\ML IMP\Simple_Linear_Regression\Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values    


from sklearn.cross_validation import train_test_split
X_train , X_test , Y_train, Y_test = train_test_split(X, Y , test_size = 1/3 , random_state= 0)
 
    


from sklearn.linear_model import LinearRegression
linearregression = LinearRegression()
linearregression.fit(X_train,Y_train)


pred_y = linearregression.predict(X_test)

plt.scatter(X_train,Y_train)
plt.scatter(X_test,Y_test, c= 'Black',marker='*')
plt.plot(X_train,linearregression.predict(X_train),c='red')
plt.show()



    