#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#create data 
df = pd.DataFrame({'cases': [898, 1055, 613, 254, 467, 1054, 1663, 2909],
                  'hospitalizations': [90, 82, 56, 27, 56, 92, 161, 206], 
                  'deaths': [4, 1, 2, 2, 2, 2, 7, 8]})


# In[3]:


#view data
df


# In[4]:


import statsmodels.api as sm


# In[5]:


#define response variable
y = df['deaths']


# In[7]:


#define predictor variables
x = df[['cases', 'hospitalizations']]


# In[8]:


#add constant to predictor variables
x = sm.add_constant(x)


# In[9]:


#fit linear regression model
model = sm.OLS(y, x).fit()


# In[10]:


#view model summary
print(model.summary())


# In[12]:


#import necessary libraries 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[13]:


#fit simple linear regression model
model = ols('cases ~ deaths', data=df).fit()


# In[14]:


#view model summary
print(model.summary())


# In[16]:


#define figure size
fig = plt.figure(figsize=(12,8))

#produce regression plots
fig = sm.graphics.plot_regress_exog(model, 'deaths', fig=fig)


# In[ ]:




