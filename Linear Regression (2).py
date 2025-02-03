#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("NewspaperData.csv")
data


# In[16]:


data.info()


# In[17]:


data.isnull().sum()


# In[18]:


data.describe()


# In[19]:


#Boxplot for daily column
plt.figure(figsize=(6,3))
plt.title("Box plot for Daily Sales")
plt.boxplot(data["daily"], vert = False)
plt.show()


# In[20]:


sns.histplot(data['daily'], kde = True,stat='density',)
plt.show()


# In[21]:


sns.boxplot(x=data['sunday'])
plt.title("Boxplot of Sunday Data")
plt.show()


# # Observations
# - There are no missing values
# - The daily column values appears to be right-skewed
# - The sunday column values also appear to be right-skewed
# - There are two outliers in both daily column and also in sunday column as observed from the boxplots# 

# # Scatter plot and Correlation Strength

# In[22]:


x = data["daily"]
y = data["sunday"]
plt.scatter(data["daily"], data["sunday"])
plt.xlim(0, max(x) + 100)
plt.ylim(0, max(y) + 100)
plt.show()


# In[10]:


data["daily"].corr(data["sunday"])


# In[11]:


data[["daily","sunday"]].corr()


# # Observations
# - The relationship between x (daily) and y (sunday) is seen to be linear as seen from scatter plot
# - The correlation is strong positive with Pearson's correlation coefficient of 0.958154

# # Fit a Linear regression model

# In[12]:


import statsmodels.formula.api as smf  

# Assuming 'data' is a pandas DataFrame containing the columns 'sunday' and 'daily'
model1 = smf.ols("sunday ~ daily", data=data).fit()


# In[13]:


model1.summary()


# In[ ]:




