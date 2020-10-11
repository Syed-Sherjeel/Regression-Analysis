#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Polynomial Regression
# 
# Estaimted time needed: **15** minutes
# 
# ## Objectives
# 
# After completing this lab you will be able to:
# 
# -   Use scikit-learn to implement Polynomial Regression
# -   Create a model, train,test and use the model
# 

# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="#download_data">Downloading Data</a></li>
#         <li><a href="#polynomial_regression">Polynomial regression</a></li>
#         <li><a href="#evaluation">Evaluation</a></li>
#         <li><a href="#practice">Practice</a></li>
#     </ol>
# </div>
# <br>
# <hr>
# 

# ### Importing Needed packages
# 

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# <h2 id="download_data">Downloading Data</h2>
# To download the data, we will use !wget to download it from IBM Object Storage.
# 

# In[2]:


get_ipython().system('wget -O FuelConsumption.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')


# ## Understanding the Data
# 
# ### `FuelConsumption.csv`:
# 
# We have downloaded a fuel consumption dataset, **`FuelConsumption.csv`**, which contains model-specific fuel consumption ratings and estimated carbon dioxide emissions for new light-duty vehicles for retail sale in Canada. [Dataset source](http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64?cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ&cm_mmc=Email_Newsletter-_-Developer_Ed%2BTech-_-WW_WW-_-SkillsNetwork-Courses-IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork-20718538&cm_mmca1=000026UJ&cm_mmca2=10006555&cm_mmca3=M12345678&cvosrc=email.Newsletter.M12345678&cvo_campaign=000026UJ)
# 
# -   **MODELYEAR** e.g. 2014
# -   **MAKE** e.g. Acura
# -   **MODEL** e.g. ILX
# -   **VEHICLE CLASS** e.g. SUV
# -   **ENGINE SIZE** e.g. 4.7
# -   **CYLINDERS** e.g 6
# -   **TRANSMISSION** e.g. A6
# -   **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
# -   **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
# -   **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
# -   **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0
# 

# ## Reading the data in
# 

# In[3]:


df = pd.read_csv("FuelConsumption.csv")

# take a look at the dataset
df.head()


# Lets select some features that we want to use for regression.
# 

# In[4]:


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)


# Lets plot Emission values with respect to Engine size:
# 

# In[5]:


plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()


# #### Creating train and test dataset
# 
# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set.
# 

# In[6]:


msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# <h2 id="polynomial_regression">Polynomial regression</h2>
# 

# Sometimes, the trend of data is not really linear, and looks curvy. In this case we can use Polynomial regression methods. In fact, many different regressions exist that can be used to fit whatever the dataset looks like, such as quadratic, cubic, and so on, and it can go on and on to infinite degrees.
# 
# In essence, we can call all of these, polynomial regression, where the relationship between the independent variable x and the dependent variable y is modeled as an nth degree polynomial in x. Lets say you want to have a polynomial regression (let's make 2 degree polynomial):
# 
# $y = b + \\theta_1  x + \\theta_2 x^2$
# 
# Now, the question is: how we can fit our data on this equation while we have only x values, such as **Engine Size**? 
# Well, we can create a few additional features: 1, $x$, and $x^2$.
# 
# **PloynomialFeatures()** function in Scikit-learn library, drives a new feature sets from the original feature set. That is, a matrix will be generated consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, lets say the original feature set has only one feature, _ENGINESIZE_. Now, if we select the degree of the polynomial to be 2, then it generates 3 features, degree=0, degree=1 and degree=2: 
# 

# In[7]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
train_x_poly


# **fit_transform** takes our x values, and output a list of our data raised from power of 0 to power of 2 (since we set the degree of our polynomial to 2).
# 
# $
# \\begin{bmatrix}
#     v_1\\
#     v_2\\
#     \\vdots\\
#     v_n
# \\end{bmatrix}
# $
# $\\longrightarrow$
# $
# \\begin{bmatrix}
#     [ 1 & v_1 & v_1^2]\\
#     [ 1 & v_2 & v_2^2]\\
#     \\vdots & \\vdots & \\vdots\\
#     [ 1 & v_n & v_n^2]
# \\end{bmatrix}
# $
# 
# in our example
# 
# $
# \\begin{bmatrix}
#     2\.\\
#     2.4\\
#     1.5\\
#     \\vdots
# \\end{bmatrix}
# $
# $\\longrightarrow$
# $
# \\begin{bmatrix}
#     [ 1 & 2. & 4.]\\
#     [ 1 & 2.4 & 5.76]\\
#     [ 1 & 1.5 & 2.25]\\
#     \\vdots & \\vdots & \\vdots\\
# \\end{bmatrix}
# $
# 

# It looks like feature sets for multiple linear regression analysis, right? Yes. It Does. 
# Indeed, Polynomial regression is a special case of linear regression, with the main idea of how do you select your features. Just consider replacing the  $x$ with $x_1$, $x_1^2$ with $x_2$, and so on. Then the degree 2 equation would be turn into:
# 
# $y = b + \\theta_1  x_1 + \\theta_2 x_2$
# 
# Now, we can deal with it as 'linear regression' problem. Therefore, this polynomial regression is considered to be a special case of traditional multiple linear regression. So, you can use the same mechanism as linear regression to solve such a problems. 
# 
# so we can use **LinearRegression()** function to solve it:
# 

# In[8]:


clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly, train_y)
# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)


# As mentioned before, **Coefficient** and **Intercept** , are the parameters of the fit curvy line. 
# Given that it is a typical multiple linear regression, with 3 parameters, and knowing that the parameters are the intercept and coefficients of hyperplane, sklearn has estimated them from our new set of feature sets. Lets plot it:
# 

# In[9]:


plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")


# <h2 id="evaluation">Evaluation</h2>
# 

# In[10]:


from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y) )


# <h2 id="practice">Practice</h2>
# Try to use a polynomial regression with the dataset but this time with degree three (cubic). Does it result in better accuracy?
# 

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1>Import necessary libraries</h1>
# </div>

# In[12]:


# Import libraries
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn import pipeline


# Double-click **here** for the solution.
# 
# <!-- Your answer is below:
# from sklearn import linear_model
# from sklearn import model_selection
# from sklearn import metrics
# from sklearn import preprocessing
# from sklearn import pipeline
# 
# -->

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1>Split the Dataset</h1>
# </div>

# In[13]:


#Split the dataset
X_train,X_test,y_train,y_test=model_selection.train_test_split(df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']],df[['CO2EMISSIONS']],test_size=0.3)


# Double-click **here** for the solution.
# 
# <!-- Your answer is below:
# X_train,X_test,y_train,y_test=model_selection.train_test_split(df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']],df[['CO2EMISSIONS']],test_size=0.2)
# 
# -->

# <!-- Your answer is below:
# 
# -->

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1>Standard scaling</h1>
# </div>

# In[16]:


Xtrain=preprocessing.StandardScaler().fit_transform(X_train)
Xtest=preprocessing.StandardScaler().fit_transform(X_test)


# Hint
# <!-- Your answer is below:
# X_train_poly=preprocessing.StandardScaler().fit_transform(X_train)
# X_test_poly=preprocessing.StandardScaler().fit_transform(X_test)
# -->

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1>Initiliaze polynomial features and fit them to train and test explanatory variable</h1>
# </div>

# In[17]:


#Polynomial Feature
Xtrain=preprocessing.PolynomialFeatures(degree=4).fit_transform(Xtrain)
Xtest=preprocessing.PolynomialFeatures(degree=4).fit_transform(Xtest)


# Hint
# <!-- Your answer is below:
# poly=preprocessing.PolynomialFeatures(degree=3)
# X_train_poly=poly.fit_transform(X_train)
# X_test_poly=poly.fit_transform(X_test)
# -->

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1>Initialize model Check Model performance</h1>
# </div>

# In[18]:


regr=linear_model.LinearRegression()
regr.fit(Xtrain,y_train)
yhat=regr.predict(Xtest)
metrics.r2_score(y_test,yhat),metrics.mean_squared_error(y_test,yhat)


# Hint
# <!-- Your answer is below:
# y_hat=regr.predict(X_test_poly)
# metrics.mean_squared_error(y_hat,y_test)
# metrics.r2_score(y_hat,y_test)
# -->

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1>Find the slope and intercept of the model?</h1>
# </div>

# In[19]:


#fit the mode and check intercepts and coefficents
regr.intercept_,regr.coef_


# Hint
# <!-- Your answer is below:
# regr=linear_model.LinearRegression()
# regr.fit(X_train_poly,y_train)
# regr.coef_,regr.intercept_
# 
# -->

# <!-- Your answer is below:
# X_train,X_test,y_train,y_test=model_selection.train_test_split(df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']],df[['CO2EMISSIONS']],test_size=0.2)
# 
# -->

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1> 
# Perform Kfold Cross Validation
# </h1>
# </div>

# In[22]:


model_selection.cross_val_predict(regr,Xtest,y_test,cv=4).mean()


# Hint
# <!-- Your answer is below:
# model_selection.cross_val_predict(regr,X_test, y_test,cv=4)
# -->

# <div class="alert alert-danger alertdanger" style="margin-top: 20px">
# <h1>Develop a pipeline</h1>
# </div>

# In[28]:


Input1=[('Standardize',preprocessing.StandardScaler()),
        ('PolynomialFeatures',preprocessing.PolynomialFeatures(degree=4)),
        ('Model',linear_model.Ridge(alpha=10))
       ]
Input2=[('Polynomial',preprocessing.PolynomialFeatures(degree=4)),
       ('Standardize',preprocessing.StandardScaler()),
       ('Model',linear_model.Ridge(alpha=10))]
pipe1=pipeline.Pipeline(Input1)
pipe2=pipeline.Pipeline(Input2)
pipe1.fit(X_train,y_train)
pipe2.fit(X_train,y_train)
yhat1=pipe1.predict(X_test)
yhat2=pipe1.predict(X_test)
metrics.r2_score(y_test,yhat1),metrics.r2_score(y_test,yhat2)


# Hint
# <!-- Your answer is below:
# input=[('Standardize',preprocessing.StandardScaler()),('Polynomial',preprocessing.PolynomialFeatures(degree=4)),('Model',linear_model.Ridge(alpha=.005))]
# pipe=pipeline.Pipeline(input)
# pipe.fit(X_train,y_train)
# y_hat=pipe.predict(X_test)
# metrics.r2_score(y_hat,y_test)
# -->

# 
# 
