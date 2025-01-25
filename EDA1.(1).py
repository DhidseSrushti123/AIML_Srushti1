#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("data_clean.csv")
data


# In[3]:


data.info()


# In[4]:


#data structure
print(type(data))
print(data.shape)


# In[5]:


data.shape


# In[6]:


#data Type
data.dtypes


# In[7]:


#Drop dupplication column and unnames column
data1 = data.drop(['Unnamed: 0',"Temp C"],axis =1)
data1


# In[8]:


data1.info()


# In[9]:


#Convert the month column data type to float data type
data1['Month']=pd.to_numeric(data['Month'],errors='coerce')
data1.info()


# In[10]:


#print all duplication rows
data1[data1.duplicated(keep = False)]


# In[13]:


#change column names(Rename the colunms)
data1.rename({'Solar.R':'Solar'},axis=1, inplace = True)
data1


# In[14]:


data1.info()


# In[15]:


#Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[23]:


#Visualize data1 missing values using graph
cols = data1.columns
colours = ['black','pink','green','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[24]:


#Find the mean and median values of each numeric column
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[25]:


#Replace the ozone missing values with median values
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[27]:


data1['Solar'] = data1['Solar'].fillna(median_ozone)
data1.isnull().sum()


# In[29]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ",median_solar)
print("Mean of Solar: ",mean_solar)


# In[ ]:




