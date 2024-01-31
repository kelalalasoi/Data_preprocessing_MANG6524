# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 04:39:37 2023

@author: Kelsie Nguyen
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load your dataframe
data = pd.read_excel('E:\Study Abroad\Semester 01\MANG6524 - Marketing Analytics and Visualisations\Final Assessment\Output\Eliminate noise\Df for k_means.xlsx')

# Set the maximum number of clusters
max_k = 10

# Set the label column
label = 'Country'

# Separate the numeric and categorical columns 
numeric_data = data.select_dtypes(include=[np.number])
categorical_data = data.select_dtypes(exclude=[np.number])

# Scale the numeric data
scaler = StandardScaler()
scaled_numeric_data = scaler.fit_transform(numeric_data)

# Combine the scaled numeric data and categorical data
data_scaled = pd.concat([pd.DataFrame(scaled_numeric_data, columns=numeric_data.columns), categorical_data], axis=1)

# Create a dictionary to store the optimal number of clusters for each category
optimal_k_dict = {}

# Iterate over each category
for cat in data[label].unique():
    # Extract the data for the current category
    cat_data_numeric = data_scaled[data[label] == cat].select_dtypes(include=[np.number])
    
    # Initialize a list to store the silhouette scores
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(cat_data_numeric)
        silhouette_scores.append(silhouette_score(cat_data_numeric, kmeans.labels_))
    
    # Find the optimal number of clusters
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    
    # Plot the silhouette scores for each category
    plt.figure(figsize=(10,5))
    plt.plot(range(2, max_k+1), silhouette_scores)
    plt.title(f'Silhouette Scores for Category {cat}')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    # Store the optimal number of clusters for each category
    optimal_k_dict[cat] = optimal_k
