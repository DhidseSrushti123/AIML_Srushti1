#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('diabetes.csv')
df


# In[3]:


X = df.drop('class', axis=1)
Y = df['class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,Y, test_size = 0.2, random_state = 42)


# In[10]:


gbc = GradientBoostingClassifier(random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample':  [0.2, 1.0]
}
grid_search = GridSearchCV(estimator=gbc,
                          param_grid=param_grid,
                          cv=kfold,
                          scoring='recall',
                          n_jobs=-1,
                          verbose=1)


# In[11]:


grid_search.fit(X_train, Y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Recall:", grid_search.best_score_)


# In[14]:


best_model = grid_search.best_estimator_
Y_pred = best_model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))


# In[ ]:




