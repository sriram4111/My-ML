# Basic LinearRegression Model
# with one independent feature and one dependent feature Weight vs Height.
# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

#read and load the data the csv file 
df = pd.read_csv("/content/Height_Weight.csv")

#it will get top 5 rows
df.head()

#check how many rows and columns is there
df.shape

#Scatter plot use to plot my dataset data visually
plt.scatter(df['Weight'],df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")

#check Correlation blw weight and height
 df.corr()

#Seaborn for visualizations
import seaborn as sns
sns.pairplot(df)

#independent features and dependent features
X = df[['Weight']]#--->Independent feature has to be in dataframe or 2d array
y = df['Height']

X=df[["Weight"]]
np.array(X).shape

np.array(y).shape

#Train_Test_split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)

X_train.shape

#standadization-->datapreprocessing

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)

X_train

X_test = scaler.transform(X_test)

X_test

#Apply Linear Regression model
from sklearn.linear_model import LinearRegression

regression  = LinearRegression(n_jobs=-1)

regression.fit(X_train,y_train)

#get coeff_ and Intercept_
print("Coefficent or slope:",regression.coef_)
print("Intercept:",regression.intercept_)

##plot my training data to get best_fit line
plt.scatter(X_train,y_train) #plot dataset points, True data points of X_train and y_train
plt.plot(X_train,regression.predict(X_train)) #create best fit line, predicted points based on X_train data

#prediction for test_data
#my model has to predict for given new data ##X_test having new data
y_pred = regression.predict(X_test)

#Performance metrics 
#I have to know how much my model actual accuracy us we use Performance metrics
from sklearn.metrics import mean_squared_error,mean_absolute_error

mse = mean_squared_error(y_test,y_pred)
mae= mean_absolute_error(y_test,y_pred)
rmse = np.sqrt(mse)
print(mse)
print(mae)
print(rmse)

#R-square
from sklearn.metrics import r2_score

score = r2_score(y_test,y_pred)
print(score)

#display adjusted Rsquare
Adjusted_R = 1-(1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print(Adjusted_R)

#OLS Linear Regression
import statsmodels.api as sm

model = sm.OLS(y_train,X_train).fit()
pred = model.predict(X_test)
print(pred)

print(model.summary())

#predict the new data
regression.predict([[72]]) ###So you have to use scaler.transform to get values of height right else you will get large values
#like height 856.20Centimeters
##Output : array([856.20893941])

#predict the new data #When we use scaler.transform it gives me good height data..!
regression.predict(scaler.transform([[72]]))
##Output : array([163.27367587])

regression.predict(scaler.transform([[74]]))
#Output : array([164.28717785])