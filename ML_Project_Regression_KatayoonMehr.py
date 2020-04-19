# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:17:03 2020

@author: Katayoon Mehr
"""

########################################################################

### Continuious y -  Regression ###

''' Linear Regression, KNN, adaboost, Random Forest, SVR '''

import numpy as np
import pandas as pd
import os
os.chdir('C:\\Users\\Rayan\\Desktop\\Kati\\Machine Learning\\Project\\Continuous')
#os.chdir('E:\\Machine Learning\\Project')
os.getcwd()

House=pd.read_csv('House_Pr.csv', delimiter=',')

House.head()
House.describe()
House.columns
House.info()
House.isnull().any()
House.isnull().sum()

# Handling Missing and Encoding is NOT needed


y = House['price']
X = House[['bedrooms','bathrooms','sqft_living', 'sqft_lot', \
           'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',\
           'yr_built']]

import seaborn as sns
import matplotlib.pyplot as plt
X_Corr = X.corr()
sns.heatmap(X.corr(), cmap="YlGnBu")


plt.hist(y, bins = 50)
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Distribution of Price")



import statistics 

print("Median of price : % s "
        % (statistics.median(House.price))) 

print("Mean of price : % s "
        % (statistics.mean(House.price))) 

print("Standard Deviation of sample is % s " 
       % (statistics.stdev(House.price)))


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train_S=sc.fit_transform(X_train)
X_test_S=sc.transform(X_test)



''' Linear Regression '''

from sklearn.linear_model import LinearRegression
model= LinearRegression()

model.fit(X_train_S, y_train)
y_pred_lr = model.predict(X_test_S)

print(model.intercept_)
print(model.coef_)

import statsmodels.api as sm
X2_train_S=sm.add_constant(X_train_S)
ols = sm.OLS(y_train, X_train_S)
lr=ols.fit()
print(lr.summary())

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred_lr)




''' KNN '''

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, recall_score, \
                         precision_score, confusion_matrix, classification_report
  

y = House['price']
X = House[['bedrooms','bathrooms','sqft_living', 'sqft_lot', 'floors',\
           'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',\
           'yr_built']]
                       
model = KNeighborsRegressor(n_neighbors=10, metric = 'minkowski', weights = 'uniform')

model.fit(X_train_S, y_train)
classifier.score(X_test, y_test)
y_pred_knn = model.predict(X_test_S)


score=[]
for i in range (1, 20):
    classifier = KNeighborsRegressor(n_neighbors = i, weights='uniform')
    classifier.fit(X_train, y_train)
    sc=classifier.score(X_test, y_test)
    score.append(sc)

import matplotlib.pyplot as plt
plt.plot(range(1,20), score)



''' Adaboost ''' 

from sklearn.ensemble import AdaBoostRegressor

ada = AdaBoostRegressor(n_estimators=3, random_state=10)
ada.fit(X_train_S, y_train)
ada.score(X_test_S, y_test)

y_pred_ada = ada.predict(X_test_S)
 

scores=[]
for i in [2,3,4,5,10, 20]:
    ada = AdaBoostRegressor(n_estimators=i, random_state=10)
    ada.fit(X_train_S, y_train)
    scores.append(ada.score(X_test_S, y_test))  
    
import matplotlib.pyplot as plt
plt.plot([2,3,4,5,10, 20], scores)




''' Random Forest '''

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=1000, random_state=0)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)
y_pred_rf = rf.predict(X_test)

  
scores=[]
for i in [10, 100, 1000, 2500]:
    rf = RandomForestRegressor(n_estimators=i, random_state=10)
    rf.fit(X_train, y_train)
    scores.append(rf.score(X_test, y_test))   
    
import matplotlib.pyplot as plt
plt.plot([10, 100, 1000, 2500], scores)
plt.xlabel("no of modelss")
plt.ylabel("Accuracy on the test dataset")




''' SVR '''

from sklearn.svm import SVR
                         
svr = SVR(kernel = 'poly', gamma = 5 , C=1)
svr.fit(X_train_S, y_train)
svr.score(X_test_S, y_test)
y_pred_svr = svr.predict(X_test_S)


from sklearn.model_selection import GridSearchCV
param_dict = {
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma' : [0.1, 1, 5],
                'C': [1, 10]                
            }

grid = GridSearchCV(SVR(), param_dict, cv=4)
grid.fit(X_train_S, y_train)
grid.best_params_
grid.best_score_







