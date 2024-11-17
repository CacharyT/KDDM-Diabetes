#Visualization specifically for Target and Early Onset Symtomps

cluster_sample = sampled_df

#Encoding Target and early onset sympstoms
target_map = target_mapping
early_onset_map = {'Yes': 1, 'No': 0}

cluster_sample['Target_encoded'] = cluster_sample['Target'].map(target_map)
cluster_sample['early_onset_symptoms_encoded'] = cluster_sample['Early Onset Symptoms'].map(early_onset_map)

features = ['Target_encoded', 'early_onset_symptoms_encoded']
data = cluster_sample[features].values

centroids_s, clusters_s = k_means(data, 3)

# Plotting the results
plt.figure(figsize=(8, 6))

# Scatter plot of the data points, colored by clusters
scatter = plt.scatter(cluster_sample['Target_encoded'], cluster_sample['early_onset_symptoms_encoded'], c=clusters, cmap='viridis', marker='o', alpha=0.6)

# Plot the centroids in red with a larger 'X' marker
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

# Adding labels and title
plt.title("K-Means Clustering with Target and Early Onset Symptoms")
plt.xlabel('Target')
plt.ylabel('Early Onset Symptoms')
plt.legend()
plt.colorbar(scatter, label='Cluster ID')
plt.show()