#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train=pd.read_csv('D:/archive/train.csv')
train


# In[3]:


train.head()


# # EXPLORATORY DATA ANALYSIS

# # missing data

# In[4]:


train.isnull()


# In[5]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

roughly 20% of age data is missing from the dataset.The proportion of missing age data is likely small enough than the cabin missing data which is roughly more than 95%.
# In[6]:


sns.set_style("whitegrid")
sns.countplot(x='Survived',data=train)


# In[7]:


plt.title("sex survival count")
sns.set_style("whitegrid")
sns.countplot(x='Survived',hue='Sex',data=train,palette="Set3")

This graph interperates that the count of men deaths are far more than women 
deaths, hecne the count of women survival are more than men 
# In[8]:


plt.title("passanger class survival")
sns.set_style("whitegrid")
sns.countplot(x='Survived',hue='Pclass',data=train,palette="rainbow")


# In[9]:


sns.displot(train['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[10]:


plt.title("count of siblings and spouse")
sns.set_style("whitegrid")
sns.countplot(x='SibSp',data=train)


# In[11]:


plt.title("count of siblings and spouse according to passenger class ")
sns.set_style("whitegrid")
sns.countplot(x='SibSp',hue='Pclass',data=train)


# In[12]:


plt.figure(figsize=(12,7))
plt.title("predcting the average age in each passanger class")
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')


# # Removing the null values from dataset

# In[13]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass ==2:
            return 29
        elif Pclass==3:
            return 24
    else:
        return Age


# In[14]:


train['Age']=train[['Age','Pclass']].apply(impute_age,axis=1)


# In[15]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

so as by the heatmap the null values from  the age data has removed
# In[16]:


train.drop('Cabin',axis=1,inplace=True)
train


# In[17]:


train.head()


# In[18]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # converting categorial features

# In[19]:


train.describe()


# In[20]:


train.info()


# In[21]:


pd.get_dummies(train['Embarked'],drop_first=True).head()


# In[22]:


sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)


# In[23]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[24]:


train.head()

train=pd.concat([train,sex,embark],axis=1)
# In[25]:


train.head()


# # Building a Logistic Regression model
# 

# # Train Test Split

# In[26]:


train.drop('Survived',axis=1).head()


# In[27]:


train['Survived'].head()


# In[29]:


from sklearn.model_selection import train_test_split


# In[35]:


X_train,X_test,Y_train,Y_test=train_test_split(train.drop('Survived',axis=1),
                                               train['Survived'],test_size=0.30,
                                               random_state=101)


# # Training and predicting

# In[36]:


from sklearn.linear_model import LogisticRegression


# In[37]:


logmodel=LogisticRegression()
logmodel.fit(X_train,Y_train)


# In[38]:


predictions=logmodel.predict(X_test)


# In[39]:


from sklearn.metrics import confusion_matrix


# In[40]:


accuracy=confusion_matrix(Y_test,predictions)


# In[41]:


accuracy


# In[42]:


from sklearn.metrics import accuracy_score


# In[43]:


accuracy=accuracy_score(Y_test,predictions)
accuracy


# In[44]:


predictions


# In[ ]:




