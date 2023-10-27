import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
# Create a sample DataFrame
data = [[ 1,0,4],
        [-1,1,4],
        [ 0,2,4],
        [ 2,1,3]]
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

data = np.array(data).T
def pca(data, num_components):
    # Center the data
    centered_data = np.array(data - np.mean(data, axis=0))

    # Compute the covariance matrix
    cov_matrix = np.dot(centered_data, centered_data.T) / (len(data) - 1)

    # Perform eigendecomposition on the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top-k eigenvectors based on the desired number of components
    selected_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Transform the data to the new coordinate system
    transformed_data = np.dot(selected_eigenvectors.T,centered_data )
    return transformed_data

transformed_data = pca(data,2)
print(transformed_data)
for row in transformed_data.T:
    ax.scatter(row[0],row[1])
plt.show()