#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


iris = pd.read_csv("iris.csv")


# In[3]:


iris


# In[4]:


# Bar plot
import seaborn as sns
counts = iris["variety"].value_counts()
sns.barplot(data = counts)


# In[5]:


iris.info()


# In[6]:


iris[iris.duplicated(keep= False)]


# In[8]:


counts = iris["variety"].value_counts()
plt.bar(counts.index,counts.values)


# # Observations
# - There are 150 rows and 5 columns
# - There are no null values
# - There is one duplicated row
# - The x-columns are sepal.lenghth,sepal.width,petal.length and petal.width
# - All the x-columns are continous
# - The y-column is variety which is categorical

# In[9]:


# Drop the duplication
iris = iris.drop_duplicates(keep='first')


# In[10]:


iris[iris.duplicated]


# In[12]:


# Rest the index
iris = iris.reset_index(drop=True)
iris


# # Perform label encoding of target column

# In[15]:


# Encode the three flower classes as 0,1,2

labelencoder = LabelEncoder()
iris.iloc[:, -1] = labelencoder.fit_transform(iris.iloc[:,-1])
iris.head()


# In[14]:


iris.info()


# # Observation
# - The target column('variety')is still object type.It needs to be converted to numeric"(int)

# In[16]:


# Convert the target column data to integer

iris['variety'] = pd.to_numeric(labelencoder.fit_transform(iris['variety']))
print(iris.info())


# In[17]:


# Divide the dataset in to x-columns and y-columns
X=iris.iloc[:,0:4]
Y=iris['variety']


# In[19]:


# Further splitting of data into training and testing data sets
x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,random_state = 1)
x_train


# # Building Decision Tree Classifier Using Entropy Criteria

# In[20]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth =None)
model.fit(x_train,y_train)


# In[24]:


# plot the decision tree
plt.figure(dpi=1200)
tree.plot_tree(model);


# In[25]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
plt.figure(dpi=1200)
tree.plot_tree(model,feature_names = fn, class_names=cn,filled = True);


# In[ ]:




