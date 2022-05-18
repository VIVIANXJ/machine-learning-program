#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from numpy import asarray


# In[3]:


df = pd.read_csv("Assignment 2 dataset - data1.csv")


# In[23]:


from sklearn.model_selection import train_test_split
X = np.array(df)


# In[5]:


from sklearn.preprocessing import MinMaxScaler


# In[26]:


from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=200, centers=4,
                       cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=100);


# In[ ]:




