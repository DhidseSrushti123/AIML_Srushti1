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


# In[2]:


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

# In[6]:


cars[cars.duplicated()]


# # Pair plots and correlation coefficents

# In[7]:


sns.set_style(style='darkgrid')
sns.pairplot(cars)


# #### Observations
#  - Extreme values observed towards the right tail.
#  - A few outliers present in both tails.
#  - Outliers may be due to unique high-performance car models.

# In[8]:


cars.corr()


# # Observations from correlation plots and Coefficients
# - Between x and y,all the x variables are showing moderate to high correlation strengths, highest being between HP and MPG
# - Therefore this dataset qualifies for building a multiple linear regression model to predict MPG
# - Among x columns among x columns is not desirable as it might lead to multicollinearity problem

# # Preparing a preliminary model considering all xcolumns

# In[9]:


# Build model
model = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()


# In[10]:


model.summary()


# # Observation from  modelsummary
# 
# - The R-square and adjusted R-squared values are good and about 75% of variability in Y is explained by X columns
# - The probabiltity values with respect to F- statics is close to zero, indication that all or some of X columns are significant
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves, which need to be further explored

# # Performance metrics for model1

# In[11]:


df1 = pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[12]:


pred_y1 = model.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[13]:


# Compute the Mean squared Error (MSE) for model1
from sklearn.metrics import mean_squared_error
print("MSE :", mean_squared_error(df1["actual_y1"], df1["pred_y1"]))


# In[16]:


from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(df1["actual_y1"], df1["pred_y1"])
print("MSE", mse)
print("RMSE :", np.sqrt(mse))


# # Checking for multicollinearity among X-columns using VIF method

# In[18]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# # Observations:
# 
# - The ideal range of VIF values shall be between 0 to 10. However slightly higher values can be tolerated
# - As seen from the very high VIF values for VOL and WT , itis clear that they are prone to multicollinearity problem
# - Hence it is decided to drop one of the columns(either VOL or Wt) to overcome the multicollinearity.
# - It is decided to drop WT and retain VOL column in further models

# In[19]:


cars1 = cars.drop("WT", axis=1)
cars1.head()


# In[20]:


# Build model 
# Import statsmodel.formula.api as smf
model2 = smf.ols('MPG~VOL+SP+HP',data=cars1).fit()


# In[21]:


model2.summary()


# In[22]:


# Find the performance metrics
df2 = pd.DataFrame()
df2["actual_y2"] = cars["MPG"]
df2.head()


# In[23]:


pred_y2 = model2.predict(cars.iloc[:,0:4])
df2["pred_y2"] = pred_y2
df2.head()


# In[29]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2["pred_y2"])
print("MSE :", mse )
print("RMSE :",np.sqrt(mse))


# ## Observations from model2 summary
# - The adjusted R-suared value improved slightly to 0.76
# - All the p-values for model parameters  are less than 5% hence they are significant
# - Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG response varible
# - There is no improvement in MSE value

# In[30]:


cars1


# In[ ]:




