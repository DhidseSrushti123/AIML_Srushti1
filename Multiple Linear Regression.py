#!/usr/bin/env python
# coding: utf-8

# Assumptions in Multilinear Regression
# 
# 1. Linear: The relationship between the predictors(x) and the response(y) is linear.
# 
# 2. Independence: Observation are independent of each other .
# 
# 3. Homoscedasticity: The residuals(Y_hat) exhibit constant variance at all level of the precictor.
#     
# 4. Normal Distribution of Eorror: The residuals of the model are normally distributed.
#     
# 5. No multicollinearity: The independent     

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot


# In[7]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[8]:


# Rearramge the columns
cars = pd.DataFrame(cars, columns=["HP","VOL","SP","WT","MPG"])
cars.head()


# # Description of columns
# 
# - MPG: Milege of the cars(Mile per Gallon)(This is Y-column to be predicated)
# - HP: Horse Power of the car(x1 column)
# - SP: Top speed of the car (Miles per Hours)(x3 column)
# - WT: Weight of the car (pounds)(x4 column)
# - VOL: Voulme of the car(size)(x2 column) 

# In[9]:


cars.info()


# In[10]:


cars.isna().sum()


# # observations
# - The are no missing values
# - There are 81 observation (81 different cars data)
# - The data types of the columns are also relevant and valid

# In[ ]:




