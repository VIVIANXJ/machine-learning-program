#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sys


# In[2]:


df = pd.read_csv("breast-cancer-wisconsin.csv")


# In[3]:


df.iloc[20:25]


# In[4]:


df = df.replace(to_replace = {'?': np.nan, 'class1':0, 'class2':1}, value = None)


# In[5]:


df.iloc[20:25]


# In[6]:


from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')


# In[7]:


imp_mean.fit(df)


# In[8]:


df_array = imp_mean.transform(df)


# In[9]:


from sklearn.preprocessing import MinMaxScaler


# In[10]:


scaler = MinMaxScaler()


# In[11]:


df_array_norm = scaler.fit_transform(df_array)


# In[12]:


x = df_array_norm[:,:9]
y = np.array(list(map(int,df_array_norm[:,-1])))


# In[13]:


x_new = []
for i in range(len(x)):
    line = []
    sample = x[i]
    label = y[i]
    for data in sample:
        line.append(np.round(data,4))
        print("%.4f, "%(data),end = '')
    x_new.append(line)
    print(label)
x_new = np.array(x_new)


# In[14]:


from sklearn.model_selection import StratifiedKFold


# In[15]:


cvKFold=StratifiedKFold(n_splits=10, shuffle=True, random_state=0)


# In[16]:


train_in = []
test_in = []
for train_index, test_index in cvKFold.split(x_new, y):
    train_in.append(train_index)
    test_in.append(test_index)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier


# In[18]:


def KNNClassifier(X, y, k):
    #define classifier
    neigh = KNeighborsClassifier(n_neighbors=k)
    scores = []
    #10 cross
    for train_index, test_index in cvKFold.split(x_new, y):
        #get dataset based on index
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        # put dataset into KNNobject
        neigh.fit(x_train,y_train)
        score = neigh.score(x_test, y_test)
        scores.append(score)
    
    scores = np.array(scores)   
    
    return scores, scores.mean()


# In[19]:


KNNClassifier(x_new,y,4)


# In[20]:


from sklearn.linear_model import LogisticRegression


# In[21]:


def logregClassfier(x,y):
    clf = LogisticRegression(random_state=0)
    scores = []
    for train_index, test_index in cvKFold.split(x_new, y):
        
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        clf.fit(x_train,y_train)
        score = clf.score(x_test,y_test)
        scores.append(score)
    scores = np.array(scores)
    return scores,scores.mean()


# In[22]:


logregClassfier(x_new,y)


# In[23]:


from sklearn.naive_bayes import GaussianNB


# In[24]:


def nbClassifier(X, y):
    clf = GaussianNB()
    scores = []
    for train_index, test_index in cvKFold.split(x_new, y):
        
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        clf.fit(x_train,y_train)
        score = clf.score(x_test,y_test)
        scores.append(score)
    scores = np.array(scores)
    return scores,scores.mean()


# In[25]:


nbClassifier(x_new,y)


# In[26]:


from sklearn.tree import DecisionTreeClassifier


# In[27]:


def dtClassifier(X, y):
    
    clf = DecisionTreeClassifier(random_state=0)
    scores = []
    for train_index, test_index in cvKFold.split(x_new, y):
        
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        clf.fit(x_train,y_train)
        score = clf.score(x_test,y_test)
        scores.append(score)
    scores = np.array(scores)
    return scores,scores.mean()


# In[28]:


dtClassifier(x_new, y)


# In[29]:


from sklearn.ensemble import BaggingClassifier


# In[30]:


def bagDTClassifier(X, y, n_estimators, max_samples, max_depth):    
    clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=0), n_estimators=500,
    max_samples=100, bootstrap=True, random_state=0)
    scores = []
    for train_index, test_index in cvKFold.split(x_new, y):
        
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        clf.fit(x_train,y_train)
        score = clf.score(x_test,y_test)
        scores.append(score)
    scores = np.array(scores)
    return scores,scores.mean()


# In[31]:


bagDTClassifier(x_new, y, 100, 100, 2)


# In[32]:


from sklearn.ensemble import AdaBoostClassifier


# In[33]:


def adaDTClassifier(X, y, n_estimators, learning_rate, max_depth):
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    scores = []
    for train_index, test_index in cvKFold.split(x_new, y):
        
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        clf.fit(x_train,y_train)
        score = clf.score(x_test,y_test)
        scores.append(score)
    scores = np.array(scores)
    return scores,scores.mean()


# In[34]:


adaDTClassifier(x_new, y,100,0.2,3)


# In[35]:


from sklearn.ensemble import GradientBoostingClassifier


# In[36]:


def gbClassifier(X, y, n_estimators, learning_rate):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, random_state=0)
    scores = []
    for train_index, test_index in cvKFold.split(x_new, y):
        
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        clf.fit(x_train,y_train)
        score = clf.score(x_test,y_test)
        scores.append(score)
    scores = np.array(scores)
    return scores,scores.mean()


# In[37]:


gbClassifier(x_new,y,100,0.2)


# In[38]:


from sklearn.svm import LinearSVC
def bestLinClassifier(X,y):
    clf= SVC(kernel="linear")
    clf.fit(x_train,y_train)


# In[39]:



param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
def bestLinClassifier(X,y):
    clf= SVC(kernel="linear")
    for train_index, test_index in cvKFold.split(x_new, y):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        clf.fit(x_train,y_train)
        grid_search = GridSearchCV(SVC(), param_grid, cv=10,
                          return_train_score=True)
        grid_search.fit(x_train,y_train)
    print("{:.4f}".format(grid_search.best_params_['C']))
    print("{:.2f}".format(grid_search.best_params_['gamma']))
    print("{:.4f}".format(grid_search.score(x_test, y_test)))
    print("{:.4f}".format(grid_search.best_score_))


# In[40]:


bestLinClassifier(x_new,y)


# In[41]:


from sklearn.ensemble import RandomForestClassifier

param_grid = {'n_estimators': [10, 20, 50, 100],
              'max_features': ['auto','sqrt','log2'],
              'max_leaf_nodes':[10, 20, 30]}
def bestRFClassifier(X,y):
    rnd_clf = RandomForestClassifier(n_estimators=100, max_features="auto", max_leaf_nodes=10, random_state=0)
    
    for train_index, test_index in cvKFold.split(x_new, y):
        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        rnd_clf.fit(x_train,y_train)
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid)
        grid_search.fit(x_train,y_train)
        params=grid_search.best_params_
        
        #score = rnd_clf.score(x_test,y_test)
        print(params['n_estimators'])
        print(params['max_features'])
        print(params['max_leaf_nodes'])
        print('{:.4f}'.format(grid_search.best_score_))
        print('{:.4f}'.format(grid_search.score(x_test,y_test)))


# In[42]:


bestRFClassifier(x_new, y)


# In[46]:





# In[ ]:





# In[ ]:




