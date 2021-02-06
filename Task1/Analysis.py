#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('notebook', 'matplotlib')
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import plotly.graph_objects as go
import plotly.express as px
import plotly 
import matplotlib.pyplot as plt
import re
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D


# In[6]:


df_purchase = pd.read_csv("./QVI_purchase_behaviour.csv")
df_purchase.head()


# In[7]:


df_purchase.info()


# In[8]:


df_purchase.shape


# In[9]:


print(df_purchase["PREMIUM_CUSTOMER"].nunique())
print(df_purchase["LIFESTAGE"].nunique())


# In[10]:


df_transaction = pd.read_excel("QVI_transaction_data.xlsx")
df_transaction.head()


# In[11]:


df_transaction.info()


# In[12]:


df_transaction.shape


# In[71]:


df_analyze = df_transaction.merge(df_purchase,how="left",on="LYLTY_CARD_NBR").dropna()


# In[72]:


df_analyze.shape


# In[73]:


df_analyze.head()


# In[74]:


df_Life_sales = df_analyze[["TOT_SALES","LIFESTAGE"]].groupby("LIFESTAGE").agg(np.sum)
df_Life_sales


# In[75]:


# Before detecting anomalies
plt.plot(df_Life_sales)

plt.tick_params(axis="x", labelrotation=90)
plt.tick_params(axis="y", labelrotation=0)

plt.figure;


# In[76]:


sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=df_analyze["TOT_SALES"], y=df_analyze["LIFESTAGE"])


# In[77]:


# Remove outliers
def reject_outliers(sr, iq_range=0.995):
    pcnt = (1 - iq_range) / 2
    qlow, median, qhigh = sr.quantile([pcnt, 0.50, 1-pcnt])
    iqr = qhigh - qlow
    return sr[ (sr - median).abs() <= iqr]

df_analyze["TOT_SALES"]  = reject_outliers(df_analyze["TOT_SALES"], 0.9)
df_analyze["TOT_SALES"]


# In[78]:


sns.set_theme(style="dark")
ax = sns.boxplot(x=df_analyze["TOT_SALES"], y=df_analyze["LIFESTAGE"])


# In[79]:


df_status_sales = df_analyze[["TOT_SALES","PREMIUM_CUSTOMER"]].groupby("PREMIUM_CUSTOMER").agg(np.sum)
df_status_sales


# In[80]:


sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=df_analyze["TOT_SALES"], y=df_analyze["PREMIUM_CUSTOMER"])


# In[81]:


df_life_status_sales = df_analyze[["TOT_SALES","PREMIUM_CUSTOMER","LIFESTAGE"]].groupby(["PREMIUM_CUSTOMER","LIFESTAGE"]).agg(np.sum)
df_life_status_sales


# In[82]:


df_life_status_sales.index


# In[83]:



sns.set(style = "darkgrid")

fig = sns.factorplot(x="LIFESTAGE", y='TOT_SALES',data= df_analyze[["TOT_SALES","PREMIUM_CUSTOMER","LIFESTAGE"]],
                     kind='bar', col="PREMIUM_CUSTOMER")
fig.set_xlabels('');


# In[84]:


sns.set_theme(style="whitegrid")
ax = sns.boxplot( x=df_analyze["PROD_QTY"])


# In[85]:


df_analyze["PROD_QTY"]  = reject_outliers(df_analyze["PROD_QTY"], 0.995)
df_analyze["PROD_QTY"]


# In[86]:


sns.set_theme(style="whitegrid")
ax = sns.boxplot(x=df_analyze["PROD_QTY"])


# In[87]:


df_analyze.shape


# In[88]:


df_analyze = df_analyze.dropna()


# In[89]:


df_analyze.shape


# In[90]:


df_analyze["PROD_NAME"]


# In[91]:


pd.to_numeric(df_analyze['DATE']);


# In[98]:


pd.to_datetime(df_analyze['DATE'])
df_analyze.head()


# In[93]:


df_analyze = df_analyze.dropna()


# In[97]:


# df_analyze["DATE"] = pd.to_datetime(df_analyze["DATE"], origin = '1899-12-30').dt.date
df_analyze["DATE"]


# In[46]:


df_analyze['LYLTY_CARD_NBR'] = df_analyze['LYLTY_CARD_NBR'].astype('str')
df_analyze['TXN_ID'] = df_analyze['TXN_ID'].astype('str')
df_analyze['STORE_NBR'] = df_analyze['STORE_NBR'].astype('str')
df_analyze['PROD_NBR'] = df_analyze['PROD_NBR'].astype('str')


# In[47]:


fig = px.histogram(df_analyze, x="LIFESTAGE", y="TOT_SALES", color="PROD_QTY",
                   marginal="box",
                   hover_data=df_analyze.columns)
fig.show()


# In[64]:


df_sales_p_d = df_analyze[["DATE","TOT_SALES"]].groupby("DATE").agg(np.sum)
df_sales_p_d


# In[ ]:




