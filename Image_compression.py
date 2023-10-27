from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Read the JPG image using PIL
image = Image.open('test_img.jpg')

# Convert the image to a NumPy array
image_array = np.array(image)

REDUCE_TO=50

# Flatten the image array
flattened_array =(image_array.reshape(-1, image_array.shape[-1]) )
# Perform PCA to reduce dimensionality
mean = flattened_array.mean()

standardlize_matrix = np.array(flattened_array - mean)

cov_matrix = np.dot(standardlize_matrix,standardlize_matrix.T) / len(standardlize_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]

eigenvectors = eigenvectors[:, idx]
selected_eigenvectors = eigenvectors[:,:REDUCE_TO]


new_coordinates = np.dot(selected_eigenvectors.T, standardlize_matrix)

print(new_coordinates.shape)