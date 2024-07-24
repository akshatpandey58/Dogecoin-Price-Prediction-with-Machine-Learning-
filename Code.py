#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor 


# In[2]:


data = pd.read_csv(r'C:\Users\Acer\OneDrive\Documents\DOGE-USD.csv') 
data.head() 


# In[3]:


data.corr()


# In[4]:


data['Date'] = pd.to_datetime(data['Date'], 
							infer_datetime_format=True) 
data.set_index('Date', inplace=True) 

data.isnull().any() 


# In[5]:


data.isnull().sum() 


# In[6]:


data = data.dropna()


# In[7]:


data.describe() 


# In[8]:


plt.figure(figsize=(20, 7)) 
x = data.groupby('Date')['Close'].mean() 
x.plot(linewidth=2.5, color='b') 
plt.xlabel('Date') 
plt.ylabel('Volume') 
plt.title("Date vs Close of 2021") 


# In[9]:


data["gap"] = (data["High"] - data["Low"]) * data["Volume"] 
data["y"] = data["High"] / data["Volume"] 
data["z"] = data["Low"] / data["Volume"] 
data["a"] = data["High"] / data["Low"] 
data["b"] = (data["High"] / data["Low"]) * data["Volume"] 
abs(data.corr()["Close"].sort_values(ascending=False)) 


# In[10]:


data = data[["Close", "Volume", "gap", "a", "b"]] 
data.head() 


# In[11]:


df2 = data.tail(30) 
train = df2[:11] 
test = df2[-19:] 

print(train.shape, test.shape) 


# In[12]:


from statsmodels.tsa.statespace.sarimax import SARIMAX 
model = SARIMAX(endog=train["Close"], exog=train.drop( 
	"Close", axis=1), order=(2, 1, 1)) 
results = model.fit() 
print(results.summary()) 


# In[13]:


start = 11
end = 29
predictions = results.predict( 
	start=start, 
	end=end, 
	exog=test.drop("Close", axis=1)) 
predictions 


# In[14]:


test["Close"].plot(legend=True, figsize=(12, 6)) 
predictions.plot(label='TimeSeries', legend=True) 


# In[ ]:




