import matplotlib
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from imblearn.over_sampling import SMOTE


matplotlib.use('TkAgg')

# Generate synthetic data
X2, y2 = make_blobs(n_samples = [20, 200], centers = [[1, 1], [4, 4]], cluster_std = [2.5, 1.0], random_state = 14)
data2 = pd.DataFrame(X2, columns = ["X1", "X2"])
data2["Class"] = y2

# Initialize the figure and axis
fig, ax = plt.subplots()
plt.subplots_adjust(bottom = 0.25)

# Initial number of neighbors
initial_neighbors = 5

# Plot the data
ax.scatter(data2[data2["Class"] == 1]["X1"], data2[data2["Class"] == 1]["X2"], label = "Majority Class", marker = "o")
ax.scatter(data2[data2["Class"] == 0]["X1"], data2[data2["Class"] == 0]["X2"], label = "Minority Class", marker = "o", alpha = 1)
smote_scatter = ax.scatter([], [], label = "SMOTE", marker = "x", alpha = 0.5)
ax.set_title("Resampled Data using SMOTE")
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.legend(loc = "upper left")

# Create the slider
axcolor = 'lightgoldenrodyellow'
ax_neighbors = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor = axcolor)
s_neighbors = Slider(ax_neighbors, 'Neighbors', 1, 15, valinit = initial_neighbors, valstep = 1)

# Update function for the slider
def update(val):
    neighbors = int(s_neighbors.val)
    smote = SMOTE(k_neighbors = neighbors)
    X_resampled, y_resampled = smote.fit_resample(data2[["X1", "X2"]], data2["Class"])
    smote_scatter.set_offsets(X_resampled[y_resampled == 0])
    ax.legend(loc = "upper left")
    fig.canvas.draw_idle()

# Attach the update function to the slider
s_neighbors.on_changed(update)

# Explicitly use plt.show() with block=False to activate the event loop
plt.show(block = False)
plt.pause(0.1)
plt.show()
