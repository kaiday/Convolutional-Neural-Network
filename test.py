import matplotlib.pyplot as plt
import numpy as np

# Define the vectors
v1 = np.array([1, 1, 1])
v2 = np.array([-1, 2, 0])

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the vectors
origin = np.array([0, 0, 0])
ax.quiver(*origin, *v1, color='red')
ax.quiver(*origin, *v2, color='blue')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the plot limits
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

# Show the plot
plt.show()