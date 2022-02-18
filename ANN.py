# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:47:06 2022

@author: prasc
"""

# Load Libraries Step 1
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from IPython.display import display, Math, Latex
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from hydroeval import *
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten

import seaborn as sns
#%matplotlib inline


# Set working directory
os.chdir('C:\\Users\\prasc\\Desktop\\Concrete Compression Strength')


# Read the dataset (Step 3)
#data = pd.read_csv('concretedata.csv') # read our dataset
#data = pd.read_excel('concretedata.xlsx', usecols = range(9))

## 1. Perform exploratory data analysis (EDA) on the multivariate (concrete compression) dataset provided to you. You can make new variables by combining data from different attributes if necessary. Things to consider here include summary statistics, correlations, empirical cumulative distribution functions, scatter plots between inputs and outputs, etc.
# Reading data, creating features / labels
data = pd.read_excel('concretedata.xlsx', usecols = range(9))
X, y = np.transpose(data.values[:,:-1]), data.values[:,-1]
x_labels, y_label = list(data.columns[:8]), data.columns[-1]

# Descriptive Statistics
print (data.describe())
data.describe(include='all')

# Correlations between features
Correlations = []
for i in range(7):
    for j in range(i+1, 8):
        Correlations += ['Correlation({x},{y})={ro}'.format(x = x_labels[i],
                                                  y = x_labels[j],
                                                  ro = str(round(np.corrcoef(X[i], X[j])[1][0], 2)))]
    
print (sorted(Correlations, key = lambda x: abs(float(x.split('=')[-1])))[::-1])

# Correlations between features and label
Correlations_label = []
for i in range(7):
    Correlations_label += ['Correlation({x},{y})={ro}'.format(x = x_labels[i],
                                                  y = y_label,
                                                  ro = str(round(np.corrcoef(X[i], y)[1][0], 2)))]
    
print (sorted(Correlations_label, key = lambda x: abs(float(x.split('=')[-1])))[::-1])


pairplot=sns.pairplot(data,palette="Set2") # creating pairwise plot

# Cumulative distribution plot
fig, axs = plt.subplots(3, 3, figsize = (24, 20))
def distribution_plot(ax, X, label):
    
    # calculating the distribution
    ys, xs = np.histogram(X, bins = 20)
    
    # plotting the cumulative distribution
    ax.plot(xs[:-1], np.cumsum(ys), 
            color = 'orange',
            linewidth = 3, 
            label = label)

    ax.grid()
    ax.legend(fontsize = 23, loc = 'lower right')


for i in range(8):
    distribution_plot(axs[i//3][i%3], X[i], x_labels[i])

distribution_plot(axs[2][2], y, y_label)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.subplots_adjust(hspace = 0.15)
plt.subplots_adjust(wspace = 0.15)
plt.savefig('CumulativeDistributions.png', dpi = 125)

# Scatter plots between dimensions of X (features)
fig, axs = plt.subplots(7, 4, figsize = (24, 40))
def scatter_plot(ax, X, Y, label_x, label_y):
    # Plotting the scatter plot
    ax.scatter(X, Y, alpha = 0.35, color = 'orange')
    # Adding regression line
    m, b = np.polyfit(X, Y, deg = 1)
    x = np.hstack((np.arange(min(X), max(X)), max(X)))
    y = m*x + b
    ax.plot(x, y, color = 'firebrick', linewidth = 3.25, 
            linestyle = '--', label = r'$\rho=$'+str(round(np.corrcoef(X, Y)[1][0], 2)))
    # labels, grid, legend
    ax.set_xlabel(label_x, fontsize = 20)
    ax.set_ylabel(label_y, fontsize = 20)
    ax.grid()
    ax.legend(fontsize = 23, loc = 'upper right')


c = 0
for i in range(7):
    for j in range(i+1, 8):
        scatter_plot(axs[c//4][c%4], X[i], X[j], x_labels[i], x_labels[j])
        c+=1
        
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.subplots_adjust(hspace = 0.25)
plt.subplots_adjust(wspace = 0.25)
plt.savefig('FeatureCorrelations.png', dpi = 125)

# Scatter plots between dimensions of X (features)
fig, axs = plt.subplots(2, 4, figsize = (24, 10))
def scatter_plot(ax, X, Y, label_x, label_y):
    # Plotting the scatter plot
    ax.scatter(X, Y, alpha = 0.35, color = 'orange')
    # Adding regression line
    m, b = np.polyfit(X, Y, deg = 1)
    x = np.hstack((np.arange(min(X), max(X)), max(X)))
    y = m*x + b
    ax.plot(x, y, color = 'firebrick', linewidth = 3.25, 
            linestyle = '--', label = r'$\rho=$'+str(round(np.corrcoef(X, Y)[1][0], 2)))
    # labels, grid, legend
    ax.set_xlabel(label_x, fontsize = 20)
    ax.set_ylabel(label_y, fontsize = 20)
    ax.grid()
    ax.legend(fontsize = 23, loc = 'upper right')


for i in range(8):
    scatter_plot(axs[i//4][i%4], X[i], y, x_labels[i], y_label)
        
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.subplots_adjust(hspace = 0.25)
plt.subplots_adjust(wspace = 0.25)
plt.savefig('LabelCorrelations.png', dpi = 125)


## 2. Split the data set into training (75%) and testing (25%) using a random seed of 10 for reproducibility.
# Train-test split
X = np.transpose(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


## 3. Identify important input attributes using either Random Forest regression and/or gradient boosting regression (bonus points for using both). Think of final selection beyond what these algorithms suggest. Some points to ponder include – physical meaning of attributes, multicollinearity among variables, etc.
# Training random forest model
forest = RandomForestRegressor(n_estimators = 100,
                               n_jobs = -1,
                               oob_score = True,
                               bootstrap = True,
                               random_state = 10)
forest.fit(X_train, y_train)

print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \n'.format(forest.score(X_train, y_train),
                                                                 forest.oob_score_))
print (forest.feature_importances_)

# Plotting feature importance histogram
fig, ax = plt.subplots(figsize = (7,5))
plt.bar(range(8), height=forest.feature_importances_, color = 'royalblue')
plot = plt.xticks(np.arange(8), x_labels, rotation = 90, fontsize = 15)
plot = plt.title('Feature importance. Random Forest', fontsize = 20)
plt.yticks(fontsize = 12)
plt.grid()
plt.savefig('FeatureImportanceForest.png', dpi = 125)

# Training XGboost model
dtrain = xgb.DMatrix(X, label = y)
watchlist = [(dtrain, 'train')]
param = {'max_depth': 6, 'learning_rate': 0.03, 'random_state' : 10}
num_round = 200
bst = xgb.train(param, dtrain, num_round, watchlist, verbose_eval = False)


#print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \n'.format(bst.score(X_train, y_train),forest.oob_score_))
#print (forest.feature_importances_)


# Collecting feature importance and plotting
importance = bst.get_score(importance_type='gain')
fig, ax = plt.subplots(figsize = (7,5))
plt.bar(range(8), height=np.vectorize(importance.get)(sorted(importance.keys())), color = 'royalblue')
plot = plt.xticks(np.arange(8), x_labels, rotation = 90, fontsize = 15)
plot = plt.title('Feature importance. XGboost', fontsize = 20)
plt.yticks(fontsize = 12)
plt.grid()
plt.savefig('FeatureImportanceXGboost.png', dpi = 125)

## 4. Perform a linear regression as a baseline model for comparison. Use 5 fold cross-validation with the training data and negative root mean square error as accuracy metric.
# -RMSE metrics
def negativeRMSE(y, yhat):
    return -np.sqrt(np.mean((y-yhat)**2))

# 5-fold linear regression evaluation
score = []
observations = []
predictions = []
for i in range(5):
    clf = linear_model.LinearRegression()
    X_tr = np.roll(X_train, round(1/5*i*len(X_train)))[:round(len(X_train)*0.75)]
    y_tr = np.roll(y_train, round(1/5*i*len(y_train)))[:round(len(y_train)*0.75)]
    X_tst = np.roll(X_test, round(1/5*i*len(X_test)))[:round(len(X_test)*0.75)]
    y_tst = np.roll(y_test, round(1/5*i*len(y_test)))[:round(len(y_test)*0.75)]
    clf.fit(X_tr, y_tr)
    observations+=[y_tst]
    predictions+=[clf.predict(X_tst)]
    score+= [negativeRMSE(predictions[-1], observations[-1])]
print ('Mean negative RMSE =', str(round(np.mean(score), 4)))


## 5. Compare k-fold accuracies; significance of model coefficients and performance on testing data using Kling-Gupta Efficiency metrics (Kling et al., 2012). Check out the python package hydroeval to make this computation
# Collecting KGE values
kges = []
for i in range(5):
    kges+= [evaluator(kge, observations[i], predictions[i])]
print ('Mean KGE score =', str(round(np.mean(kges), 4)))

# Plotting KGE vs -RMSE
fig, ax = plt.subplots(figsize = (7,5))
plt.title('Kling-Gupta Efficiency vs -RMSE', fontsize = 20)
plt.yticks(fontsize = 12)
plt.plot(np.mean(kges, axis = 1), label = 'Average KGE')
plt.plot(score, label = 'negative RMSE')
plt.legend(fontsize = 20)
plt.plot()
plt.grid()
plt.savefig('KGE-RMSE.png', dpi = 125)


## 6. Develop a deep neural network model. See Young et al., 2019 for some ideas on the architecture. Use the same training and testing data; perform 5 fold cross-validation and provide the same set of metrics used for linear regression.
NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(8, kernel_initializer='normal', input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(16, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(16, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(16, kernel_initializer='normal', activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

# 5-fold validation of the Deep model
score = []
observations = []
predictions = []
for i in range(5):
    clf = linear_model.LinearRegression()
    X_tr = np.roll(X_train, round(1/5*i*len(X_train)))[:round(len(X_train)*0.75)]
    y_tr = np.roll(y_train, round(1/5*i*len(y_train)))[:round(len(y_train)*0.75)]
    X_tst = np.roll(X_test, round(1/5*i*len(X_test)))[:round(len(X_test)*0.75)]
    y_tst = np.roll(y_test, round(1/5*i*len(y_test)))[:round(len(y_test)*0.75)]
    NN_model.fit(X_tr, y_tr, epochs = 500, batch_size = 32, validation_split = 0.2)#, callbacks = callbacks_list)
    observations+=[y_tst]
    predictions+=[NN_model.predict(X_tst)]
    score+= [negativeRMSE(predictions[-1], observations[-1])]

print ('Deep Neural Network: Mean negative RMSE =', str(round(np.mean(score), 4)))

