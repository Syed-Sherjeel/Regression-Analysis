#!/usr/bin/env python
# coding: utf-8

# # Introduction
#    In this project our sole aim is to model relationship between Body mass index(BMI) and Life expectancy of data from various countries and its source is GapMinder.For this purpose we will use  a statistic technique known as linear Regression.Body Mass index or in short BMI is a value derived from the mass and height of a person. The BMI is defined as the body mass divided by the square of the body height, and is universally expressed in units of kg/m², resulting from mass in kilograms and height in metres.BMI has major impact on   life expectancy as it impacts health of human being in term of obesity and underweight etc.
#          
#    For most adults, an ideal BMI is in the 18.5 to 24.9 range. For children and young people aged 2 to 18,  the BMI calculation takes into account age and gender as well as height and weight. If your BMI is: below 18.5 – you're in the underweight range.
# # Linear Regression

#    Linear regression is a statistical method of modelling relationship between an independent variable(s) and dependent variable.  
#     __In statistic__,linear regression is a technique used to model relation between scalar response and one or more (depending upon situation) explanatory variable.Essentially linear regression outputs a continous values depending upon values of one or more parameters.
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
# <img src="https://iq.opengenus.org/content/images/2018/08/d1-1.png" alt="Drawing" style="width: 370px;"/>
# 

# where $theta o$ is y intercept that is it is the point where our randomly initialized line intecept y axis where as $theta 1$ is slop is the value that determines wether our line is going upward or downward or straight.Essentially, it determines the direction of our line or randomly initialized hypothesis.And our main goal is to minimize above cost function which is simply the difference between our predicted value and actual value.We want to find the value of $theta o$ and $theta 1$ which make the value of this function minimum

# <img src="https://miro.medium.com/max/900/1*G3evFxIAlDchOx5Wl7bV5g.png" alt="Drawing" style="width: 350px;"/>
# 

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
import pandas as pd


# # Importing data

# In[3]:


bmi_data=pd.read_csv('bmi.csv')
bmi_data.head()


# # Preprocessing

# we will check for all null values and will chose either to drop null values or fill it with backward interpolation of forward interpolation.

# In[10]:


#checking null values
bmi_data.isnull()
#dropping null values
bmi_data.dropna()
#fill na value with amount equal to value next to it in same column in data set
bmi_data.fillna(method='ffill')
# fill na values with amount equal to value behind it in same column
bmi_data.fillna(method='bfill')


# ### Exploratory data analysis
# Here we will perform some exploratory data analysis to get insight from data.

# Lets check countries which have higher BMI then average BMI in our data set and countries which have lower BMI then average BMI.

# In[64]:


#Greater then average
Greater_then_average_BMI=bmi_data[bmi_data['BMI']>bmi_data['BMI'].mean()]
#Less then average
Lower_then_average_BMI=bmi_data[bmi_data['BMI']<bmi_data['BMI'].mean()]


# # Visualizing our data

# In[13]:


import matplotlib.pyplot as plt
plt.plot(bmi_data[['BMI']],bmi_data[['Life expectancy']],'*')


# # implementing linear regression to model relation between both data

# In[65]:



bmi_model=LinearRegression()
bmi_model.fit(bmi_data[['BMI']],bmi_data[['Life expectancy']])


# # Testing our trained model

# In[68]:


bmi_model.predict([[21.0931]])


# which is quiet close to our value 60.31

# # Disadvantages

# Linear regression as indicated by linear,work well when data is linear.Its performace is effected greatly when we 
# try to model a polynomial system with linear regression.For polynomial models,we can use variant of
# linear regression known as polynomial regression.In polynomial regression,our variables are no longer linear but are mixed of square 
# of variables.

# <img src="Nonlineardata.png" alt="Drawing" style="width: 350px;"/>

# Linear Regression is very sensitive to outliers.Its performance is effected greatly to outliers and this problem
# doesnot have solution in linear regression for this we can use other algorithm like random forest.
# ![]("withoutOutlier.png")

# <img src="withoutOutlier.png" alt="Drawing" style="width: 350px;"/>
#  

# # With Outlier

# <img src="withoutliers.png" alt="Drawing" style="width: 350px;"/>  
#  

# # Conclusion

# Linear regression is a great algorithm when modelling relationship between two or more quantity.It does perform
# exceptionally well and fast on linear data set.However,its performance is greatly impacted by outliers and non linear data.
