# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
import time

# %% [markdown]
# # Credit card
# #### Data loading

# %%
data = pd.read_csv('E5-Credit-Card-Users.csv')
print(data.head())
print(data.info())

# %% [markdown]
# #### Basic scatter plot to see the distrubution of the features

# %%
fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 20))
axes = axes.flatten()

for i, col in enumerate(data.columns[1:]):
    axes[i].scatter(data.index, data[col], alpha=0.5)
    axes[i].set_title(col)
    axes[i].set_xlabel('Row Number')
    axes[i].set_ylabel('Value')

plt.tight_layout()
plt.show()

# %% [markdown]
# #### Checking on the missing values

# %%
missing_percent = data.isnull().mean() * 100
missing_values = data.isnull().sum()

print("Missing Values (%):")
print(missing_percent)

print("\nNumber of Missing Values:")
print(missing_values)

# %% [markdown]
# Since the two features that contain missing value have  scatter polt conentrated at a certain area it could be go to use mean to fill them

# %%
data.fillna(data.drop(columns=['CUST_ID']).mean(), inplace=True)
print(data.info())

# %% [markdown]
# Selection of 5 features with highest varaince

# %%
variances = data.drop(columns=['CUST_ID']).var()
print("Feature Variances:")
print(variances.sort_values(ascending=False))

top5_features = variances.sort_values(ascending=False).head(5).index.tolist()
print("Top 5 features based on variance:", top5_features)

data_selected = data[top5_features]

# %% [markdown]
# #### Standardization

# %%
scaler = StandardScaler()
data_standardized_all = scaler.fit_transform(data.drop(columns=['CUST_ID']))
data_standardized_all = pd.DataFrame(data_standardized_all, columns=data.columns[1:])

data_standardized_selected = data_standardized_all[top5_features]
print(data_standardized_selected.head())

corr_matrix_selected = data_standardized_selected.corr()
print("Correlation Matrix of Selected Features:")
print(corr_matrix_selected)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_selected, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Top 5 Features")
plt.show()


# %% [markdown]
# #### Elbow method

# %%
inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_standardized_selected)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()


# %% [markdown]
# #### K-means and PCA

# %%
optimal_k = 4

start_time = time.time()
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(data_standardized_selected)
end_time = time.time()
print("K-Means Clustering Time:", end_time - start_time)

silhouette_kmeans = silhouette_score(data_standardized_selected, kmeans_labels)
print("K-Means Silhouette Score:", silhouette_kmeans)


pca = PCA(n_components=2, random_state=42)
data_pca = pca.fit_transform(data_standardized_selected)

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clusters (PCA Visualization)')
plt.colorbar(label='Cluster')
plt.show()


# %% [markdown]
# #### Hierarchical Clustering

# %%
plt.figure(figsize=(15, 7))
dendrogram = sch.dendrogram(sch.linkage(data_standardized_selected, method='ward'))
plt.title('Dendrogram (Ward Linkage)')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distances')
plt.show()

start_time = time.time()
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, affinity='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(data_standardized_selected)
end_time = time.time()
print("Hierarchical Clustering Time:", end_time - start_time)

silhouette_hierarchical = silhouette_score(data_standardized_selected, hierarchical_labels)
print("Hierarchical Clustering Silhouette Score:", silhouette_hierarchical)


# %% [markdown]
# #### DBSCAN Clustering

# %%
start_time = time.time()
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_standardized_selected)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_outliers = list(dbscan_labels).count(-1)
percent_outliers = n_outliers / len(dbscan_labels) * 100
end_time = time.time()
print("DBSCAN Clustering Time:", end_time - start_time)

print("DBSCAN found", n_clusters_dbscan, "clusters")
print("Percentage of outliers:", percent_outliers, "%")

if n_clusters_dbscan > 1:
    mask = dbscan_labels != -1
    silhouette_dbscan = silhouette_score(data_standardized_selected[mask], dbscan_labels[mask])
    print("DBSCAN Silhouette Score (excluding outliers):", silhouette_dbscan)
else:
    print("Silhouette score cannot be computed for DBSCAN with less than 2 clusters.")


# %%
data_selected = data_standardized_selected.copy()  # create a copy to avoid modifying the original data
data_selected['cluster'] = kmeans_labels

# Compute the mean values for each feature grouped by the cluster labels
cluster_means = data_selected.groupby('cluster').mean()

# Print the computed means
print("Mean values for each feature by K-Means cluster:")
print(cluster_means)

# %% [markdown]
# # Wholesale

# %%
datau = pd.read_csv('E5-UCI-Wholesale.csv')
print(datau.head())
print(datau.info())

# %%
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(datau.columns[:]):
    axes[i].scatter(datau.index, datau[col], alpha=0.5)
    axes[i].set_title(col)
    axes[i].set_xlabel('Row Number')
    axes[i].set_ylabel('Value')

plt.tight_layout()
plt.show()

# %%
missing_percent = datau.isnull().mean() * 100
missing_values = datau.isnull().sum()

print("Missing Values (%):")
print(missing_percent)

print("\nNumber of Missing Values:")
print(missing_values)

# %%
variances = datau.var()
print("Feature Variances:")
print(variances.sort_values(ascending=False))

top5_features = variances.sort_values(ascending=False).head(5).index.tolist()
print("Top 5 features based on variance:", top5_features)

datau_selected = datau[top5_features]

# %%
scaler = StandardScaler()
datau_standardized_all = scaler.fit_transform(datau.drop(columns=['Channel']))
datau_standardized_all = pd.DataFrame(datau_standardized_all, columns=datau.columns[1:])

datau_standardized_selected = datau_standardized_all[top5_features]

corru_matrix_selected = datau_standardized_selected.corr()
print("Correlation Matrix of Selected Features:")
print(corru_matrix_selected)

plt.figure(figsize=(8, 6))
sns.heatmap(corru_matrix_selected, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Top 5 Features")
plt.show()


# %%
inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(datau_standardized_selected)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()


# %%
optimal_k = 5
start_time = time.time()
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(datau_standardized_selected)
end_time = time.time()
print("K-Means Clustering Time:", end_time - start_time)

silhouette_kmeans = silhouette_score(datau_standardized_selected, kmeans_labels)
print("K-Means Silhouette Score:", silhouette_kmeans)

pca = PCA(n_components=2, random_state=42)
data_pca = pca.fit_transform(datau_standardized_selected)

plt.figure(figsize=(8, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clusters (PCA Visualization)')
plt.colorbar(label='Cluster')
plt.show()


# %%
plt.figure(figsize=(15, 7))
dendrogram = sch.dendrogram(sch.linkage(datau_standardized_selected, method='ward'))
plt.title('Dendrogram (Ward Linkage)')
plt.xlabel('Samples')
plt.ylabel('Euclidean Distances')
plt.show()
start_time = time.time()
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, affinity='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(datau_standardized_selected)
end_time = time.time()
print("Hierarchical Clustering Time:", end_time - start_time)

silhouette_hierarchical = silhouette_score(datau_standardized_all, hierarchical_labels)
print("Hierarchical Clustering Silhouette Score:", silhouette_hierarchical)

# %%
start_time = time.time()
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(datau_standardized_selected)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_outliers = list(dbscan_labels).count(-1)
percent_outliers = n_outliers / len(dbscan_labels) * 100
end_time = time.time()
print("DBSCAN Clustering Time:", end_time - start_time)

print("DBSCAN found", n_clusters_dbscan, "clusters")
print("Percentage of outliers:", percent_outliers, "%")

if n_clusters_dbscan > 1:
    mask = dbscan_labels != -1
    silhouette_dbscan = silhouette_score(datau_standardized_selected[mask], dbscan_labels[mask])
    print("DBSCAN Silhouette Score (excluding outliers):", silhouette_dbscan)
else:
    print("Silhouette score cannot be computed for DBSCAN with less than 2 clusters.")


# %%
data_selected = datau_standardized_selected.copy()  # make a copy to avoid modifying original data
data_selected['cluster'] = dbscan_labels

# Exclude outliers (which are labeled as -1)
clustered_data = data_selected[data_selected['cluster'] != -1]

# Calculate the mean values for each feature grouped by cluster
cluster_means = clustered_data.groupby('cluster').mean()

print("Mean values for each cluster (excluding outliers):")
print(cluster_means)


