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


# In[10]:


iris.isnull().sum().sum()


# In[11]:


# VISUALIZATION


# In[12]:


plt.figure(figsize=(15,8))
sns.boxplot(x='sepal_length',y='petal_length',data=df.sort_values('sepal_length',ascending=False))


# In[13]:


sns.jointplot(x="sepal_length", y="sepal_width", data=df, size=5)


# In[14]:


sns.jointplot(x="petal_length", y="petal_width", data=df, size=5)


# In[15]:


sns.pairplot(df, hue="species", height=5)


# In[16]:


from pandas.plotting import andrews_curves
andrews_curves(iris, "species")


# In[17]:


plt.figure(figsize=(15,15))
sns.catplot(x='species',y='sepal_width',data=df.sort_values('sepal_width',ascending=False),kind='boxen')


# In[18]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='petal_length',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='petal_width',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='sepal_length',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='sepal_width',data=df)


# In[19]:


# MODEL CREATION


# In[20]:


X=iris.drop('species',axis=1)
y=iris['species']


# In[21]:


from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical


# In[22]:


iris['species'] = pd.Categorical(iris.species)
iris['species'] = iris.species.cat.codes
y = to_categorical(iris.species)


# In[23]:


# MODEL TRAINING


# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,stratify=y,random_state=123)


# In[25]:


model=Sequential()
model.add(Dense(100,activation='relu',input_shape=(4,)))
model.add(Dense(3,activation='softmax'))


# In[26]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[27]:


history=model.fit(X_train,y_train,epochs=45,validation_data=(X_test, y_test))


# In[28]:


model.evaluate(X_test,y_test)


# In[29]:


pred = model.predict(X_test[:10])
print(pred)


# In[30]:


history.history['accuracy']


# In[31]:


plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'])
plt.show()

