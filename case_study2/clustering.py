# Data Processing function for this specific dataset
from prep_menC import *

# Data modelling libraries
import pandas as pd
import numpy as np

# Scaling the values so it suits for clustering
from sklearn.preprocessing import StandardScaler

# K-Means clustering
from sklearn.cluster import KMeans

# Silhouette score
from sklearn.metrics import silhouette_score


# Get the dataframe, and make it suitable for clustering
# Import the cleaned dataframe
df = get_dataframe_MENSC()

# Convert the df into matrix
X = df.as_matrix()

# Scaling Comment / Uncomment to see changes
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Random state
rs = 42


def default_cluster3():
	# Build the clustering model on default setting, print its properties

	# Set the random state then fit the data into the model
	k3_model = KMeans(n_clusters=3, random_state=rs).fit(X)

	# Sum of intra-cluster distances
	print("\nSum of intra-cluster distance:", k3_model.inertia_)

	print("\nCentroid locations:")
	for centroid in k3_model.cluster_centers_:
		print(centroid)

	# Visualsie
	visualise_pairplot(k3_model)



def visualise_pairplot(model):
	# Visualise the clustering model

	# Assign the cluster ID to each record in X
	y = model.predict(X)
	df['Cluster_ID'] = y

	# How many records are in each cluster
	print("\nCluster membership")
	print(df['Cluster_ID'].value_counts(), "\n")

	# Pairplot the cluster distribution.
	cluster_g = sns.pairplot(df, hue='Cluster_ID')
	plt.show()

#default_cluster3()




def visualise_variables(model):
	# Assign the cluster ID to each record in X
	y = model.predict(X)
	df['Cluster_ID'] = y

	# prepare the column and bin size. Increase bin size to be more specific, but 20 is more than enough
	cols = ['AnnualSales', 'SFloorSize', 'TotalInvestment']
	n_bins = 20

	# inspecting cluster 0 and 1
	clusters_to_inspect = [0,1,2]

	for cluster in clusters_to_inspect:
		# inspecting cluster 0
		print("Distribution for cluster {}".format(cluster))
		# create subplots
		fig, ax = plt.subplots(nrows=3)
		ax[0].set_title("Cluster {}".format(cluster))

		for j, col in enumerate(cols):
			# create the bins
			bins = np.linspace(min(df[col]), max(df[col]), 20)
			# plot distribution of the cluster using histogram
			sns.distplot(df[df['Cluster_ID'] == cluster][col], bins=bins, ax=ax[j], norm_hist=True)
			# plot the normal distribution with a black line
			sns.distplot(df[col], bins=bins, ax=ax[j], hist=False, color="k")

		plt.tight_layout()
		plt.show()




# Task 3 Question 1 Optimal k

def optimalK_elbow():
	# list to save the clusters and cost
	clusters = []
	inertia_vals = []

	# this whole process should take a while
	for k in range(1, 25, 1):
		# train clustering with the specified K
		model = KMeans(n_clusters=k, random_state=rs, n_jobs=10)
		model.fit(X)

		# append model to cluster list
		clusters.append(model)
		inertia_vals.append(model.inertia_)

	# plot the inertia vs K values
	plt.plot(range(1,25,1), inertia_vals, marker='*')
	plt.show()

	for k in range(1, 24, 1):
		#print("\n", clusters[k], "\n")
		print("Silhouette score for k=", k+1, silhouette_score(X, clusters[k].predict(X)))


""" Uncomment for testing
if __name__ == '__main__':
	optimalK_elbow()
"""



# Task 3 Question 3 Optimal model
# Build the clustering model based on optimal k value
def optimal_cluster():
	
	# Set the random state then fit the data into the model
	optimal_model = KMeans(n_clusters=3, random_state=rs).fit(X)

	# Sum of intra-cluster distances
	print("\nSum of intra-cluster distance:", optimal_model.inertia_)

	print("\nCentroid locations:")
	for centroid in optimal_model.cluster_centers_:
		print(centroid)


	# Visualsie
	visualise_pairplot(optimal_model)

	visualise_variables(optimal_model)

optimal_cluster()


