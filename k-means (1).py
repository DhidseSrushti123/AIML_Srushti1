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

# In[2]:


Univ = pd.read_csv("Universities.csv")
Univ


# In[3]:


Univ.describe()


# # Standardization of data

# In[4]:


Univ1 = Univ.iloc[:,1:]


# In[5]:


Univ1


# In[6]:


cols = Univ1.columns


# In[7]:


# Standardisation function
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_Univ_df = pd.DataFrame(scaler.fit_transform(Univ1),columns = cols)
scaled_Univ_df


# In[9]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[10]:


clusters_new.labels_


# In[11]:


set(clusters_new.labels_)


# In[12]:


Univ['clusterid_new'] = clusters_new.labels_


# In[13]:


Univ


# In[14]:


Univ.sort_values(by = "clusterid_new")


# In[15]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# # Observation 
# - cluster 2 apperas to be the top rated universities cluster as the cut off score,Top 10,SFRatio paparmeter mean values are highest
# - cluster 1 appears to occupy the middle level rated universities
# - cluster 0 comes as the lower rated universities

# In[16]:


Univ[Univ['clusterid_new']==0]


# # Finding optimal k values using elbow plot

# In[20]:


wcss =[]
for i in range(1, 20):
    
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    
    wcss.append(kmeans.inertia_)
print(wcss)
plt.plot(range(1, 20),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[21]:


Univ.info()


# In[ ]:




