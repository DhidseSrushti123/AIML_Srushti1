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


# In[11]:


#change column names(Rename the colunms)
data1.rename({'Solar.R':'Solar'},axis=1, inplace = True)
data1


# In[12]:


data1.info()


# In[13]:


#Display data1 missing values count in each column using isnull().sum()
data1.isnull().sum()


# In[14]:


#Visualize data1 missing values using graph
cols = data1.columns
colours = ['black','pink','green','red']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colours),cbar = True)


# In[15]:


#Find the mean and median values of each numeric column
median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ",median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[16]:


#Replace the ozone missing values with median values
data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[17]:


data1['Solar'] = data1['Solar'].fillna(median_ozone)
data1.isnull().sum()


# In[18]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ",median_solar)
print("Mean of Solar: ",mean_solar)


# In[19]:


#Find the mode values of categorical column(weather)
print(data["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[20]:


#Impute missing values (Replace NaN with mode etc.) of "WEather" using
data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[22]:


print(data["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[23]:


data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[24]:


data1.tail()


# In[25]:


data1.reset_index(drop=True)


# # Detection of outliers in the columns
# 
# # Method1:Using histograms and box plots

# In[28]:


#create a figure with two subplots,stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios':[1,3]})

sns.boxplot(data=data1["Ozone"],ax=axes[0], color='skyblue', width=0.5, orient = 'h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")


sns.histplot(data["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[0].set_title("Histogram with KDE")
axes[0].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequence")

plt.tight_layout()
plt.show()


# # Observations
# - The Ozone column has extreme values beyond 81 as seen from box plot
# - The same is confirmed from the below right-skewed histogram

# In[31]:


# Create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Violin Plot")



# In[34]:


sns.violinplot(data=data1["Solar"], color='lightpink')
plt.title("Violin Plot")


# In[ ]:




