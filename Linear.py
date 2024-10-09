# -*- coding: utf-8 -*-
"""
Basic Linear Regression Model
One independent feature and one dependent feature Experience vs Salary
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

df = pd.read_csv("/content/Exp_Sal.csv")

df.head()

df.shape

#I want to plot my data
plt.scatter(df['Experience'],df['Salary'])
plt.title("Exp Vs Sal")
plt.xlabel("Experience")
plt.ylabel("Salary")

#correlation btw my dependent and independent feature
df.corr()

#plot correlation in data visualization
sns.pairplot(df)

X = df[['Experience']]
X

Y = df['Salary']
Y

#Train and Test Data
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_train

X_test =  scaler.transform(X_test)

X_test

#Apply my Linear Regression
from sklearn.linear_model import LinearRegression

regression  = LinearRegression(n_jobs=-1)

#train my model

regression.fit(X_train,Y_train)

#To get coefficent
regression.coef_

#To get Intercept
regression.intercept_

#plot my data to create best fit line

plt.scatter(X_train,Y_train,color="Blue",label="Real_Data")
plt.scatter(X_train,regression.predict(X_train),color="red",label="Predicted_Data")

#predict my data by giving new data
y_pred = regression.predict(X_test)

print(y_pred)

regression.predict(scaler.transform([[10]]))

#I want to know how accurate my model is performing i use performance metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

mse = mean_squared_error(Y_test,y_pred)
mae = mean_absolute_error(Y_test,y_pred)
rmse = np.sqrt(mse)

print(mse)
print(mae)
print(rmse)

#Rsquare
from sklearn.metrics import r2_score

score = r2_score(Y_test,y_pred)
score

#Adjusted Rsquare
1-(1-score)*(len(Y_test-1)/(len(Y_test-1))-(X_test.shape[-1]-1))

#By using OLS Linear Regression

import statsmodels.api as sm
model = sm.OLS(Y_train,X_train).fit()

pred = model.predict(X_test)

print(pred)

print(model.summary())

