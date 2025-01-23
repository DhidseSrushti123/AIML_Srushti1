#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("Universities.csv")
df


# In[3]:


np.mean(df["SAT"])


# In[4]:


np.median(df["SAT"])


# In[5]:


df.describe()


# In[6]:


#stantdard deviation of data 
np.std(df["GradRate"])


# In[7]:


#Find the variance
np.var(df["SFRatio"])


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[10]:


sns.histplot(df["Accept"], kde =True)


# In[11]:


plt.figure(figsize=(6,3))
plt.title("GradRate")
plt.hist(df["GradRate"])


# In[12]:


plt.figure(figsize=(6,2))
plt.title("GrandRate")
plt.hist(df["SFRatio"])


# ## observations
#  - In Acceptance ratio the data distribution in non-symmertical and right skewed

# # Visualization using boxplot

# In[13]:


#Create a pandas series of batsman1 scores
s1 = [20,15.10,25,30,35,28,40,45,60]
scores1 = pd.Series(s1)
scores1


# In[14]:


plt.boxplot(scores1, vert=False)


# In[25]:


#Create a pandas series of batsman1 scores
s1 = [20,15.10,25,30,35,28,40,45,600,150,200]
scores1 = pd.Series(s1)
scores1 


# In[26]:


plt.boxplot(scores1, vert=False)


# In[27]:


import pandas as pd
import numpy as np


# In[28]:


df = pd.read_csv("Universities.csv")
df


# In[29]:


plt.boxplot(scores1, vert=False)


# In[ ]:




