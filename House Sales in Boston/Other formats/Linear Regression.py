#!/usr/bin/env python
# coding: utf-8

# #                      Multiple Linear Regession-a case study approach
# 
# 
# # Introduction
#    In this project our sole aim is to model relationship between price of houses in boston and several factor in order to under what are different factors that effect house of data and how much that factor impact our housing price.For example how much number of rooms impact the price of data or number of floors impact on data set.
#       
# # Linear Regression

#   Multiple Linear regression is a statistical method of modelling relationship between more then one independent variable and dependent variable.  
#     __In statistic__,linear regression is a technique used to model relation between scalar response and  more then one explanatory variable.Essentially linear regression outputs a continous values depending upon values of one or more parameters.
# ## Example:
#    relation between,   
#    i-fair of taxi with respect to time  
#    ii-price of house with respect to certain parameters of house

# # Background
# Sir Francis Galton, FRS was an English Victorian era statistician, polymath, sociologist, psychologist,anthropologist, eugenicist, tropical explorer, geographer, inventor, meteorologist, proto-geneticist, and psychometrician. He was knighted in 1909.
# 
# Galton produced over 340 papers and books. He also created the statistical concept of correlation and widely promoted regression toward the mean. He was the first to apply statistical methods to the study of human differences and inheritance of intelligence.
#  [However there exist a dispute details of which can be found here](http://econ.ucsb.edu/~doug/240a/The%20Discovery%20of%20Statistical%20Regression.htm)

# <img src="https://i.imgur.com/vB3UAiH.jpg" alt="Drawing" style="width: 300px;"/>

# In linear regression our main goal is to minimize this root mean square error.For this purpose in our studies we use a method called gradient descent.In gradient descent we tend compare our predicted value with actual values according to dataset and take step in a direction with learning rate alpha.
# <img src="https://media.geeksforgeeks.org/wp-content/uploads/gradiant_descent.jpg" alt="Drawing" style="width: 370px;"/>
# 

# Unlike linear regression in 2 dimensional,In multiple dimension we get a plane a hyperplane that fits our data set becuase there are high number of dimension we get plane instead of line which was case for linear regression

# ![](Highdimensionlinearregression.png)

# Below is the pictorial representation of gradient descent algorithm in action.This learning rate alpha is significant 
# If our learning rate $ alpha $ is too large we may miss the global minimum however if it is too small we might suffer from computationally expensive slow rate of convergence. 
# Good practice is to start with learning rate from 0.005 then 0.01,0.03 and so on.

# <img src="https://miro.medium.com/max/3916/1*HrFZV7pKPcc5dzLaWvngtQ.png" alt="Drawing" style="width: 350px;"/>
# 

# ### Libraries that we will be needing are following,  
# i-scikitlearn(sklearn)  
# ii-pandas

# ### Pandas 
# In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. It is free software released under the three-clause BSD license.

# ### Scikitlearn 
# Scikit-learn is a free software machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support vector machines.Major advantage of chosing this library is because its rather simple to implement regression using scikitlearn and this module is highly optimized

# In[2]:



from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


# # implementing linear regression to model relation between both data

# In[3]:



# Load the data from the boston house-prices dataset 
boston_data = load_boston()
X = boston_data['data']
y = boston_data['target']

# Make and fit the linear regression model
# TODO: Fit the model and assign it to the model variable
model = LinearRegression()
model.fit(X,y)
# Make a prediction using the model


# # Testing our trained model

# In[5]:


sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
# TODO: Predict housing price for the sample_house
prediction=model.predict(sample_house)


# In[6]:


prediction

