#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv("train.csv")


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe().T


# In[8]:


sns.heatmap(data.corr(),cmap = 'Blues')
plt.show()


# In[9]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2)
for train_indices,test_indices in split.split(data,data[["Survived","Pclass","Sex"]]):
    strat_train_set = data.loc[train_indices]
    strat_test_set = data.loc[test_indices]


# In[10]:


plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1,2,2)
strat_test_set["Survived"].hist()
strat_test_set['Pclass'].hist()



# In[11]:


strat_train_set.info()


# ## estimators

# In[12]:


from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.impute import SimpleImputer

class AgeImputer(BaseEstimator,TransformerMixin):
    
    def fit(self,X,y =None):
        return self
    
    def transform(self,X):
        imputer = SimpleImputer(strategy = 'mean')
        X['Age'] = imputer.fit_transform(X[['Age']])
        return X


# In[13]:


from sklearn.preprocessing import OneHotEncoder 

class FeatureEncoder(BaseEstimator,TransformerMixin):
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[["Embarked"]]).toarray()
        
        
        column_names=['C','S','Q','N']
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
            
        matrix = encoder.fit_transform(X[['Sex']]).toarray()
        
        column_names = ['Female','Male']
        
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        return X


# In[14]:


class FeatureDropper(BaseEstimator,TransformerMixin):
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X.drop(["Embarked","Name","Ticket","Cabin","Sex","N"], axis = 1,errors = "ignore")


# In[15]:


from sklearn.pipeline import Pipeline

Pipeline = Pipeline([("ageimputer",AgeImputer()),
                    ("featureencoder",FeatureEncoder()),
                    ("featuredropper",FeatureDropper())])


# In[16]:


strat_train_set = Pipeline.fit_transform(strat_train_set)


# In[17]:


strat_train_set


# In[18]:


strat_train_set.info()


# In[19]:


from sklearn.preprocessing import StandardScaler

X= strat_train_set.drop(["Survived"],axis = 1)
y = strat_train_set['Survived']

scaler = StandardScaler()
X_data = scaler.fit_transform(X)
y_data = y.to_numpy()


# In[20]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()

param_grid = [{
    "n_estimators":[10,100,200,500],"max_depth":[None,5,10],"min_samples_split":[2,3,4]}]

grid_search = GridSearchCV(clf,param_grid,cv = 3,scoring = "accuracy",return_train_score = True)
grid_search.fit(X_data,y_data)


# In[21]:


final_clf = grid_search.best_estimator_


# In[22]:


final_clf


# In[23]:


strat_test_set = Pipeline.fit_transform(strat_test_set)


# In[24]:


X_test = strat_test_set.drop(["Survived"],axis = 1)
y_test = strat_test_set["Survived"]

scaler = StandardScaler()
X_data_test = scaler.fit_transform(X_test)
Y_data_test = y_test.to_numpy()


# In[25]:


final_clf.score(X_data_test,Y_data_test)


# In[26]:


final_data =Pipeline.fit_transform(data)


# In[27]:


final_data


# In[28]:


X_final = final_data.drop(['Survived'],axis = 1)
y_final = final_data['Survived']

scaler = StandardScaler()
X_data_final = scaler.fit_transform(X_final)
y_data_final = y_final.to_numpy()


# In[29]:


prod_clf = RandomForestClassifier()

param_grid = [{
    "n_estimators":[10,100,200,500],"max_depth":[None,5,10],"min_samples_split":[2,3,4]}]

grid_search = GridSearchCV(prod_clf,param_grid,cv = 3,scoring = "accuracy",return_train_score = True)
grid_search.fit(X_data_final,y_data_final)


# In[30]:


prod_final_clf = grid_search.best_estimator_


# In[31]:


titanic_test_data = pd.read_csv("test.csv")


# In[32]:


final_test_data = Pipeline.fit_transform(titanic_test_data)


# In[33]:


X_final_test = final_test_data
X_final_test = X_final_test.fillna(method = "ffill")

scaler = StandardScaler()
X_data_final_test = scaler.fit_transform(X_final_test)


# In[34]:


predictions = prod_final_clf.predict(X_data_final_test)


# In[36]:


final_df = pd.DataFrame(titanic_test_data["PassengerId"])
final_df['Survived'] = predictions

final_df.to_csv("predictions.csv",index = False)


# In[ ]:




