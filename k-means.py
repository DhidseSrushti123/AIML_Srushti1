#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# # Clustering -Divide the universities in to group(clusters)
# 

# In[4]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[10]:


Univ.describe()


# # Standardization of data

# In[11]:


Univ1 = Univ.iloc[:,1:]


# In[12]:


Univ1


# In[13]:


cols = Univ1.columns


# In[14]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[ ]:




