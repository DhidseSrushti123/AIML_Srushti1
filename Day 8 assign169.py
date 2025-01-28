#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data=pd.read_csv("Day_8_sales_data.csv")
data


# In[4]:


# Extract all rows where sales are greater than 1000
sales_greater_than_1000 = data[data['Sales'] > 1000]

# Display the result
print(sales_greater_than_1000)


# In[5]:


# Find all sales records for the "East" region
sales_east_region = data[data['Region'] == 'East']

# Display the result
print(sales_east_region)


# In[6]:


# Add a new column 'Profit_Per_Unit' calculated as Profit / Quantity
data['Profit_Per_Unit'] = data['Profit'] / data['Quantity']

# Display the updated dataset
print(data[['Profit', 'Quantity', 'Profit_Per_Unit']].head())


# In[7]:


# Create a new column 'High_Sales' with Yes if Sales > 1000, else No
data['High_Sales'] = data['Sales'].apply(lambda x: 'Yes' if x > 1000 else 'No')

# Display the updated dataset
print(data[['Sales', 'High_Sales']].head())


# In[ ]:




