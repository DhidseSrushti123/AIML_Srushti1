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

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot


# In[8]:


cars = pd.read_csv("Cars.csv")
cars.head()


# In[3]:


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

# In[4]:


cars.info()


# In[5]:


cars.isna().sum()


# # observations
# - The are no missing values
# - There are 81 observation (81 different cars data)
# - The data types of the columns are also relevant and valid

# fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# 
# sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
# ax_box.set(xlabel='')
# 
# sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat="density")
# ax_hist.set(ylabel='Density')
# 
# plt.tight_layout()
# plt.show()
# 

# # Observations from boxplot and histogram
# 
# - There are some extreme values(outlier) observed in towards the right tail of SP nad HP distribution.
# - In VOL and WT columns, a few outliers are observed in both of their distributions
# - The extreme values of cars data may have come from the specially designed nature of cars
# - As the multi-dimensional data, the outliers with respect to spatial dimensions may have to be considered while buliding the regression mode

# # Checking for duplication rows

# In[10]:


cars[cars.duplicated()]


# # Pair plots and correlation coefficents

# In[11]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# #### Observations
#  - Extreme values observed towards the right tail.
#  - A few outliers present in both tails.
#  - Outliers may be due to unique high-performance car models.

# In[13]:


cars.corr()


# # Observations from correlation plots and Coefficients
# - Between x and y,all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG
# - Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# - Among x columns among x columns is not desirable as it might lead to multicollinearity problem

# # Preparing a preliminary model considering all xcolumns

# In[14]:


# Build model
model = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[15]:


model.summary()


# In[ ]:




