#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr


# In[2]:


user1 = np.array([4, 5, 2, 3, 4])
user2 = np.array([5, 3, 2, 4, 5])


# In[7]:


cosine_similarity = 1 - cosine(user1, user2)
print(f"Cosine Similarity: {cosine_similarity:.4f}")


# In[8]:


pearson_corr, _ = pearsonr(user1, user2)
print(f"Pearson Correlation Similarity: {pearson_corr:.4f}")


# In[10]:


euclidean_distance = euclidean(user1, user2)
euclidean_similarity = 1 /(1+ euclidean_distance)
print(f"Euclidean Distance Similarity: {euclidean_similarity:.4f}")


# In[13]:


import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr

ratings = np.array([
    [5, 3, 4, 4, 2],
    [3, 1, 2, 3, 3],
    [4, 3, 4, 5, 1],
    [2, 2, 1, 2, 4]
])

users = ["Raju", "John", "Ramya", "Kishore"]
df = pd.DataFrame(ratings, index=users, columns=["Bahubali","Mufasa","Interstellar","RRR","Mrs"])
print(df)

def compute_similarity(df):
    num_users = df.shape[0]
    similarity_results = []

    for i in range(num_users):
        for j in range(i + 1, num_users):
            user1, user2 = df.iloc[i], df.iloc[j]

            cos_sim = 1 - cosine(user1, user2)
            pearson_sim, _ = pearsonr(user1, user2)
            euc_dist = euclidean(user1, user2)
            euc_sim = 1 / (1 + euc_dist)

            similarity_results.append([users[i], users[j], cos_sim, pearson_sim, euc_sim])

    return pd.DataFrame(similarity_results, columns=["User 1", "User 2", "Cosine Similarity", "Pearson Correlation", "Euclidean Similarity"])

similarity_df = compute_similarity(df)
print(similarity_df)


# In[ ]:




