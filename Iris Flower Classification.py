#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTING THE REQUIRED LIBRARIES


# In[2]:


import numpy as np
import pandas as pd
import pandas.plotting
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


iris=pd.read_csv('IRIS.csv')
iris.head()


# In[4]:


iris.info


# In[5]:


iris.describe()


# In[6]:


iris.shape


# In[7]:


iris['species'].value_counts()


# In[8]:


# CHECK FOR NULL VALUES


# In[9]:


iris.isnull().sum()


# In[8]:


iris.isnull().sum().sum()


# In[11]:


# VISUALIZATION


# In[9]:


plt.figure(figsize=(15,8))
sns.boxplot(x='sepal_length',y='petal_length',data=iris.sort_values('sepal_length',ascending=False))


# In[10]:


sns.jointplot(x="sepal_length", y="sepal_width", data=iris, size=5)


# In[11]:


sns.jointplot(x="petal_length", y="petal_width", data=iris, size=5)


# In[12]:


sns.pairplot(iris, hue="species", height=5)


# In[16]:


from pandas.plotting import andrews_curves
andrews_curves(iris, "species")


# In[13]:


plt.figure(figsize=(15,15))
sns.catplot(x='species',y='sepal_width',data=iris.sort_values('sepal_width',ascending=False),kind='boxen')


# In[14]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='petal_length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='petal_width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='sepal_width',data=iris)


# In[15]:


# MODEL CREATION


# In[16]:


X=iris.drop('species',axis=1)
y=iris['species']


# In[17]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


# In[18]:


iris['species'] = pd.Categorical(iris.species)
iris['species'] = iris.species.cat.codes
y = to_categorical(iris.species)


# In[19]:


# MODEL TRAINING


# In[20]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,stratify=y,random_state=123)


# In[21]:


model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(4,)))
model.add(Dense(3,activation='softmax'))


# In[22]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[23]:


history=model.fit(X_train,y_train,epochs=45,validation_data=(X_test, y_test))


# In[24]:


model.evaluate(X_test,y_test)


# In[25]:


pred = model.predict(X_test[:10])
print(pred)


# In[26]:


history.history['accuracy']


# In[27]:


plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()


# In[ ]:




