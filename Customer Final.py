#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv(r'C:\Users\HP\Downloads\archive\Mall_Customers.csv')


# In[4]:


data.columns


# In[5]:


X=data[['Annual Income (k$)','Spending Score (1-100)']]


# In[6]:


from sklearn.cluster import KMeans


# In[38]:


X


# In[36]:


k_means=KMeans()
k_means.fit(X)


# In[39]:


k_means=KMeans()
k_means.fit_predict(X)


# In[40]:


wcss=[]
for i in range(1,11):
    k_means=KMeans(n_clusters=i)
    k_means.fit(X)
    wcss.append(k_means.inertia_)


# In[41]:


wcss


# In[14]:


plt.plot(range(1,11),wcss)
plt.title("Elbow Method in Machine Learning")
plt.xlabel("No.of Clusters")
plt.ylabel("wcss")
plt.show()


# In[15]:


X=data[['Annual Income (k$)','Spending Score (1-100)']]


# In[43]:


k_means=KMeans(n_clusters=5,random_state=42)
y_means=k_means.fit_predict(X)


# In[44]:


y_means


# In[46]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=100,c='green',label='Cluster1')
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=100,c='yellow',label='Cluster2')
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=100,c='blue',label='Cluster3')
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=100,c='red',label='Cluster4')
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=100,c='grey',label='Cluster5')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100.,c='orange')
plt.legend()


# In[47]:


plt.scatter(X.iloc[y_means==0,0],X.iloc[y_means==0,1],s=100,c='green',label='Cluster1')
plt.scatter(X.iloc[y_means==1,0],X.iloc[y_means==1,1],s=100,c='yellow',label='Cluster2')
plt.scatter(X.iloc[y_means==2,0],X.iloc[y_means==2,1],s=100,c='blue',label='Cluster3')
plt.scatter(X.iloc[y_means==3,0],X.iloc[y_means==3,1],s=100,c='red',label='Cluster4')
plt.scatter(X.iloc[y_means==4,0],X.iloc[y_means==4,1],s=100,c='grey',label='Cluster5')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=100.,c='orange')
plt.title("Customers Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()


# In[33]:


k_means.predict([[30,55]])


# In[27]:


import joblib


# In[28]:


joblib.dump(k_means,"customer_segmentation")


# In[30]:


model=joblib.load("customer_segmentation")


# In[31]:


model.predict([[50,10]])


# In[48]:


model.predict([[100,20]])


# In[ ]:




