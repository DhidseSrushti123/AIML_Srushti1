#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Install mlxtend library
get_ipython().system('pip install mlxtend')


# In[2]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt


# In[3]:


titanic = pd.read_csv("Titanic.csv")
titanic


# In[4]:


titanic.info()


# # Observation:
# - There is no null datatype in observation
# - All columns are objects datatype and categorical in nature
# - As the columns are categorical, we can adopt one-hot-ending

# In[6]:


counts = titanic['Class'].value_counts()
plt.bar(counts.index, counts.values)


# In[7]:


#Perform one hot-encoding on categorical columns
df = pd.get_dummies(titanic, dtype=int)
df.head()


# In[8]:


df.info()


# # Apriori Algorithm

# In[10]:


frequent_itemsets = apriori(df, min_support = 0.05, use_colnames=True, max_len=None)
frequent_itemsets


# In[11]:


frequent_itemsets.info()


# In[13]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules


# In[14]:


rules.sort_values(by='lift',ascending = False)


# ## Conclusions
# - Adilt Females travelling in 1st class survived most

# In[15]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[ ]:




