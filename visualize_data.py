import matplotlib.pyplot as plt
from src.create_data import create_olympic_rings, create_unconditional_olympic_rings

# Create a figure with two subplots
plt.figure(figsize=(12, 5))

# Plot 1: Conditional Olympic Rings (with labels)
# plt.subplot(1, 2, 1)
# points, labels, label_map = create_olympic_rings(n_points=5000, ring_thickness=0.25, verbose=False)
# print("points.shape", points.shape)
# print("labels.shape", labels.shape)
# print("label_map", label_map)
# plt.scatter(points[:, 0], points[:, 1], c=labels, s=1, cmap='tab10')
# plt.title('Conditional Olympic Rings\n(with labels)')
# plt.gca().set_aspect('equal', adjustable='box')

# # Plot 2: Unconditional Olympic Rings (without labels)
# plt.subplot(1, 2, 2)
points = create_unconditional_olympic_rings(n_points=5000, ring_thickness=0.25, verbose=False)
print("points.shape", points.shape)
