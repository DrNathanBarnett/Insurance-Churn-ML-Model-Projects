# -*- coding: utf-8 -*-

#The dataset that was used for this project is the Mall_Customers.csv.
#The data is present https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqazhybHlXUndtejV3b1BjUUdkWUIySXFqMWp3UXxBQ3Jtc0ttZGRNdWoxb0lBcFZqZHlKTnJHNC1GUS1GTTRhdXdlRnFTZk9NVWktc21tZDBwQmJucFpDTEZvX1FRSl9HVHJFc2ZPYUJ4RVRyS0tnRnh3QzRlREI5ZnVRX1NlNzQ0WDVkT0RHMlYtamREZVpVVG9sSQ&q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fvjchoudhary7%2Fcustomer-segmentation-tutorial-in-python%2Fdata&v=p8arik6ZyyI

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")
df.head()

df.describe()

df.isnull().sum()

df.info()

df.rename(columns={"Annual Income (k$)": "AnnualIncome", "Spending Score (1-100)":"SpendingScore"}, inplace = True)
df.head()
sns.pairplot(df[['Age', 'AnnualIncome', 'SpendingScore']])
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(df['AnnualIncome'], df['SpendingScore'], s = 50)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Annual Income vs Spending Score')
plt.show()

X = df[['AnnualIncome', 'SpendingScore']]

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,6))
plt.plot(range(1,11),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow method to determine the optimal number of clusters')
plt.show()


kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
df['Cluster'] = y_kmeans
df.head()

plt.figure(figsize=(10,6))
plt.scatter(X.iloc[:,0], X.iloc[:,1], c = y_kmeans, s = 50, cmap = 'viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:,1], c = 'red', s = 200, alpha=0.75, marker = 'X')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.show()


#age vs spending score

plt.figure(figsize=(10,6))
plt.scatter(df['Age'], df['SpendingScore'], s = 50)
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Age vs Spending Score')
plt.show()

X = df[['Age', 'SpendingScore']]

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,6))
plt.plot(range(1,11),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow method to determine the optimal number of clusters')
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
df['Cluster'] = y_kmeans
df.head()


plt.figure(figsize=(10,6))
plt.scatter(X.iloc[:,0], X.iloc[:,1], c = y_kmeans, s = 50, cmap = 'viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:,1], c = 'red', s = 200, alpha=0.75, marker = 'X')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Customer Segments')
plt.show()


#Doing the same process for 3 features


X = df[['Age', 'AnnualIncome', 'SpendingScore']]

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.figure(figsize=(10,6))
plt.plot(range(1,11),wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow method to determine the optimal number of clusters')
plt.show()

kmeans = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
df['ClusterAgeIncomeSpend'] = y_kmeans
df.head()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df['Age'], df['AnnualIncome'], df['SpendingScore'], c=df['ClusterAgeIncomeSpend'], s = 50, cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income')
ax.set_zlebel('Spending Score')
plt.title('Customer Segments based on Age, Annual Income, and Spending Score')
plt.show()



plt.scatter(X.iloc[:,0], X.iloc[:,1], c = y_kmeans, s = 50, cmap = 'viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:,1], c = 'red', s = 200, alpha=0.75, marker = 'X')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Customer Segments')

plt.show()
