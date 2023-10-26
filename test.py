import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# load the dataset
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

# convert to pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# add target to DataFrame
df['target'] = data.target

mean_of_each_collumn = df.mean()
std_of_each_collumn = df.std()

standardize_data = (df - mean_of_each_collumn) / std_of_each_collumn
covarance = np.cov(standardize_data.T, bias = 1)
eigenvalue, eigenvectors = np.linalg.eig(covarance)

n_components = 3

pca_manual = np.matmul(np.array(standardize_data),eigenvectors)

pca_manual  = pca_manual[:,:n_components]


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(pca_manual[0 ,0], pca_manual[0,1], pca_manual[0,2], c='r', marker='o', label='Class 1')
ax.scatter(pca_manual[df['target'] == 0,0], pca_manual[df['target'] == 0,1], pca_manual[df['target'] == 0,2], c='b', marker='^', label='Class 2')


plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

plt.figure(figsize=(8, 6))
plt.imshow(df.corr(), cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title("Correlation Matrix")

# Add labels to the axes
plt.xticks(np.arange(df.shape[1]), range(1, df.shape[1] + 1))
plt.yticks(np.arange(df.shape[1]), range(1, df.shape[1] + 1))

plt.show()