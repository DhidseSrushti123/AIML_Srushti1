#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


Groceries_df = pd.read_csv('Groceries_dataset.csv')
Groceries_df


# In[3]:


Groceries_df.info()


# In[4]:


print(type(Groceries_df))
print(Groceries_df.shape)


# In[5]:


Groceries_df.shape


# In[6]:


Groceries_df.dtypes


# In[7]:


Groceries_df.describe()


# In[ ]:




