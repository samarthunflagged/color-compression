import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

def load_data():
    X = np.load("data/ex7_X.npy")
    return X

def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_data_points(X, idx):
    # Define colormap to match Figure 1 in the notebook
    cmap = ListedColormap(["red", "green", "blue"])
    c = cmap(idx)
    
    # plots data points in X, coloring them so that those with the same
    # index assignments in idx have the same color
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors=c, linewidth=0.1, alpha=0.7)

def plot_kMeans_RGB(X, centroids, idx, K):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Plot each pixel in RGB space using its actual color
    ax.scatter(
        X[:, 0], X[:, 1], X[:, 2],
        facecolors=X,
        edgecolors='none',
        s=2.5,
        alpha=0.9
    )

    # Plot centroids in red with 'x' markers
    ax.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c='red',
        marker='x',
        s=500,
        linewidths=3
    )

    ax.set_xlabel('R value - Redness')
    ax.set_ylabel('G value - Greenness')
    ax.set_zlabel('B value - Blueness')
    ax.set_title("Original colors and their color clusters' centroids")

    # Optional: Add light pane color
    ax.xaxis.set_pane_color((0., 0., 0., .05))
    ax.yaxis.set_pane_color((0., 0., 0., .05))
    ax.zaxis.set_pane_color((0., 0., 0., .05))

    # Optional: Set viewing angle for better 3D perception
    # ax.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.show()

def show_centroid_colors(centroids):
    palette = np.expand_dims(centroids, axis=0)
    num = np.arange(0,len(centroids))
    plt.figure(figsize=(16, 16))
    plt.xticks(num)
    plt.yticks([])
    plt.imshow(palette)


def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # Plot the examples
    plot_data_points(X, idx)
    
    # Plot the centroids as black 'x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    
    # Plot history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("Iteration number %d" %i)


