from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Read the JPG image using PIL
image = Image.open('test_img.jpg')

# Convert the image to a NumPy array
image_array = np.array(image)

# Flatten the image array
flattened_array = image_array.reshape(-1, image_array.shape[-1])
# Perform PCA to reduce dimensionality
pca = PCA(n_components=1)  # Specify the desired number of components
reduced_array = pca.fit_transform(flattened_array)
print(reduced_array)
reconstructed_array = pca.inverse_transform(reduced_array)

# Reshape the reconstructed array to the original shape
reconstructed_image = reconstructed_array.reshape(image_array.shape)

# Display the restored image
plt.imshow(reconstructed_image.astype(np.uint8))
plt.axis('off')  # Turn off the axis labels
plt.show()