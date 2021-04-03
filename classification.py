#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv(r"C:\Users\dell\Downloads\dataR2.csv")
print(df)


# In[2]:


print(df.info())


# In[3]:


X = df[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1' ]][:].values.reshape(116, 9)
y = df[['Classification']][:].values.reshape(116, 1)


# In[4]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=0, max_depth=5)
model.fit(X,y)


# In[5]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

df = pd.read_csv(r"C:\Users\dell\Downloads\dataR2.csv")
X_train = df[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1' ]][:100].values.reshape(100, 9)
y_train = df[['Classification']][:100].values.reshape(100, 1)
X_test = df[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1' ]][100:].values.reshape(16, 9)
y_test = df[['Classification']][100:].values.reshape(16, 1)

model = DecisionTreeClassifier(random_state=0, max_depth=5)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))


# In[6]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"C:\Users\dell\Downloads\dataR2.csv")
X = df[['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1' ]][:].values.reshape(116, 9)
y = df[['Classification']][:].values.reshape(116, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

model = DecisionTreeClassifier(random_state=0, max_depth=5)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))


# In[ ]:




