import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv(r"C:\Users\Kundan\OneDrive\Desktop\prodigy\task-02-customer-segmentation\archive\Mall_Customers.csv")

# View first five rows
print(data.head())

# Select features for clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Elbow method: choose number of clusters
inertia = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# K-Means model with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
data["Cluster"] = kmeans.fit_predict(X)

# Visualize clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=data["Cluster"],
    cmap="viridis"
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color="red",
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("Customer Segments Using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

# Display clustered customers
print(data.head())

# Save final results
data.to_csv("customer_clusters.csv", index=False)