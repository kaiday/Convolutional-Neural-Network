import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

source_data = 'mnist_train.csv'

df_source = pd.read_csv(source_data)
df = ( df_source.drop('label', axis=1) ).T

#Standardlize the data
mean = df.mean(numeric_only=True)

Standardlize_data = df-mean

N=len(df)

S= ( (np.matmul( np.array(Standardlize_data) ,Standardlize_data.T)) ) / N

eigenvalues, eigenvectors = np.linalg.eig(S)
idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

B=eigenvectors[:,:3]

z = np.matmul(B.T,Standardlize_data).values

z = z.T

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(z[df_source['label'] == 1,0 ], z[df_source['label'] == 1, 1], z[df_source['label'] == 1, 2], c='r', marker='o', label='Class 1')
ax.scatter(z[df_source['label'] == 2,0 ], z[df_source['label'] == 2, 1], z[df_source['label'] == 2, 2], c='b', marker='^', label='Class 1')
ax.scatter(z[df_source['label'] == 3,0 ], z[df_source['label'] == 3, 1], z[df_source['label'] == 3, 2], c='g', marker='o', label='Class 1')
plt.show()