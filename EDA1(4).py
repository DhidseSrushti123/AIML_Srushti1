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


# In[21]:


print(data["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[22]:


data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[23]:


data1.tail()


# In[24]:


data1.reset_index(drop=True)


# # Detection of outliers in the columns
# 
# # Method1:Using histograms and box plots

# In[25]:


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

# In[26]:


# Create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='lightgreen')
plt.title("Violin Plot")



# In[27]:


sns.violinplot(data=data1["Solar"], color='lightpink')
plt.title("Violin Plot")


# In[28]:


plt.figure(figsize=(6,2))
plt.boxplot(data1["Ozone"], vert= False)


# In[29]:


#Extra outlier from boxplot for ozone column
plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"], vert=False)
[item.get_xdata() for item in boxplot_data['fliers']]


# # Method 2 for outlier detection
# - Using mu+/-3*sigma limits(sdm)

# In[30]:


data1["Ozone"].describe()


# In[31]:


mu = data1["Ozone"].describe()[1]
sigma = data1["Ozone"].describe()[2]

for x in data1["Ozone"]:
    if ((x < (mu - 3*sigma)) or (x > (mu + 3*sigma))):
        print(x)


# # Observation
# - It is observed that only two outliers are identifed using std method
# - In box plot method more no of otliers are identified
# - This is because the assumption of normallity is not satisfied in the column

# In[32]:


import scipy.stats as stats

#Create Q-Q plot
plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"], dist="norm", plot=plt)
plt.title("Q-Q Plot for Outlier Detection", fontsize=14)
plt.xlabel("Theorectical Quantiles", fontsize=12)


# # Observation from Q-Q plot
# - The data does not follow normal distribution as the data points are derviating significantly away from the red line
# - The data show a right-skewed distribution and possible outliers

# - Other visualisation that could help in the detection of outliers

# # Other visualisations that could help understand the data

# In[35]:


#Create a figure for violin plot
sns.violinplot(data=data1["Ozone"], color='orange')
plt.title("Violin Plot")
#show the plot
plt.show()


# In[40]:


sns.swarmplot(data=data1, x ="Weather", y ="Ozone",color="orange",palette="Set2",size=6)


# In[46]:


sns.stripplot(data=data1, x ="Weather", y = "Ozone",color="orange",size=6, jitter = True)


# In[47]:


sns.kdeplot(data=data1["Ozone"], fill=True, color="blue")
sns.rugplot(data=data1["Ozone"], color="black")


# In[48]:


#category wise boxplot for ozone
sns.boxplot(data = data1, x = "Weather", y="Ozone")


# In[49]:


sns.boxplot(data = data1, x = "Weather", y="Solar")


# # Correlation coefficient and pair plots

# In[51]:


plt.scatter(data1["Wind"], data1["Temp"])


# In[52]:


#Compute pearson correclation coefficient
data1["Wind"].corr(data1["Temp"])


# # It is observed that the coorelation cofficient is defined
# 

# In[ ]:




