#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTING REQUIRED LIBRARIES


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# In[3]:


# VISUALIZATIONS


# In[4]:


titanic=pd.read_csv('Titanic-Dataset.csv')
titanic.head()


# In[5]:


titanic.shape


# In[6]:


titanic.describe()


# In[7]:


titanic.info()


# In[8]:


null_columns=titanic.columns[titanic.isnull().any()]
titanic.isnull().sum()


# In[9]:


sns.set(font_scale=1)
labels = []
values = []
for col in null_columns:
    labels.append(col)
    values.append(titanic[col].isnull().sum())
ind = np.arange(len(labels))
width=0.4
fig, ax = plt.subplots(figsize=(5,5))
rects = ax.barh(ind, np.array(values), color='blue')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_ylabel("Column Names")
ax.set_title("Variables with missing values");


# In[10]:


g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="black");


# In[11]:


g = sns.FacetGrid(titanic, hue="Survived", col="Sex", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=[">", "<"]))
g.map(plt.scatter, "Fare", "Age",edgecolor="black").add_legend()
plt.subplots_adjust(top=0.6)
g.fig.suptitle('Survival by Gender , Age and Fare');


# In[12]:


titanic.Embarked.value_counts().plot(kind='bar',color="cyan", alpha=0.45)
plt.title("Passengers per boarding location");


# In[13]:


titanic.Age[titanic.Pclass == 1].plot(kind='kde')    
titanic.Age[titanic.Pclass == 2].plot(kind='kde')
titanic.Age[titanic.Pclass == 3].plot(kind='kde')
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;


# In[14]:


# MISSING VALUE IMPLEMENTATION


# In[15]:


titanic[titanic['Embarked'].isnull()]


# In[16]:


titanic["Embarked"] = titanic["Embarked"].fillna('C')


# In[17]:


titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())


# In[18]:


titanic = titanic.drop(['Cabin'],axis=1)


# In[19]:


titanic.isnull().sum()


# In[20]:


titanic = titanic.drop(['Name','Ticket'],axis=1)


# In[21]:


titanic.head()


# In[22]:


# PREDICT SURVIVAL


# In[23]:


predictors = ["Pclass","SibSp","Fare","Age","Parch"]
lr = LogisticRegression(random_state=1)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
scores = cross_val_score(lr, titanic[predictors],titanic["Survived"],scoring='f1', cv=cv)
print(scores.mean())


# In[24]:


# CONVERTING CATEGORICAL VARIABLES INTO NUMERICAL VARIABLES


# In[25]:


titanic = pd.get_dummies(titanic,columns=['Sex','Embarked'],drop_first=True,dtype=int)
titanic.head()


# In[26]:


# SEPARATION OF TEST DATA AND TRAIN DATA


# In[27]:


X = titanic.drop(['Survived'],axis=1) 
y = titanic['Survived']


# In[28]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=21)


# In[29]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)


# In[30]:


display(X_train.head())
display(X_test.head())


# In[31]:


# CREATION OF GAUSSIAN MODEL


# In[32]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)


# In[33]:


# TRAINING OF GAUSSIAN MODEL


# In[34]:


gaussian_train = round(gaussian.score(X_train, y_train) * 100, 2)
gaussian_accuracy = round(accuracy_score(Y_pred, y_test) * 100, 2)

print("Training Accuracy     :",gaussian_train)
print("Model Accuracy Score  :",gaussian_accuracy)

