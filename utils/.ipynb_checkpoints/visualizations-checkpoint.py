import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA

def visualize_data(real_data, synthetic_data, feature_1, feature_2, color_real='blue', color_synthetic='yellow'):
    """
    Visualize two-dimensional data points for real and synthetic data.

    Args:
        real_data (numpy.ndarray): Real data samples.
        synthetic_data (numpy.ndarray): Synthetic data samples.
        feature_1 (int): Index of the first feature to visualize.
        feature_2 (int): Index of the second feature to visualize.
        color_real (str): Color for real data points.
        color_synthetic (str): Color for synthetic data points.
    """
    plt.plot(real_data[:, feature_1], real_data[:, feature_2], ".", color=color_real)
    plt.plot(synthetic_data[:, feature_1], synthetic_data[:, feature_2], ".", color=color_synthetic)
    plt.legend(['Real data', 'Synthetic data'])
    plt.xlabel(f"Feature {feature_1}")
    plt.ylabel(f"Feature {feature_2}")
    plt.show()

def visualize_across_epochs(real_data, synthetic_across_epochs, feature_1, feature_2, color_real='blue', color_synthetic='yellow'):
    """
    Visualize synthetic data across epochs using animation.

    Args:
        real_data (numpy.ndarray): Real data samples.
        synthetic_across_epochs (list of numpy.ndarray): List of synthetic data samples across epochs.
        feature_1 (int): Index of the first feature to visualize.
        feature_2 (int): Index of the second feature to visualize.
        color_real (str): Color for real data points.
        color_synthetic (str): Color for synthetic data points.
    """
    x = []
    y = []
    
    fig, ax = plt.subplots()
    
    def update(frame):
        x = synthetic_across_epochs[frame][:, feature_1]
        y = synthetic_across_epochs[frame][:, feature_2]
        ax.cla()
        
        ax.set_xlim(0, 3.1)
        ax.set_ylim(0, 3.1)
        
        plt.plot(real_data[:, feature_1], real_data[:, feature_2], ".", color=color_real)
        ax.set_title(f'Epoch {frame}')
        plt.plot(x, y, ".", color=color_synthetic)
        
    ani = FuncAnimation(fig, update, frames=len(synthetic_across_epochs), interval=10, repeat=False)
    plt.show()

def visualize_pca(real_data, synthetic_data, color_real='blue', color_synthetic='yellow'):
    """
    Visualize data in two-dimensional PCA space for real and synthetic data.

    Args:
        real_data (numpy.ndarray): Real data samples.
        synthetic_data (numpy.ndarray): Synthetic data samples.
        color_real (str): Color for real data points.
        color_synthetic (str): Color for synthetic data points.
    """
    pca_real = PCA(n_components=2)
    real_result = pca_real.fit_transform(real_data)
    
    pca_synthetic = PCA(n_components=2)
    synthetic_result = pca_synthetic.fit_transform(synthetic_data)

    plt.plot(real_result[:,0], real_result[:, 1],".", color=color_real)
    plt.plot(synthetic_result[:,0], synthetic_result[:, 1],".", color=color_synthetic)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Visualization')
    plt.legend(['Real data', 'Synthetic data'])
    plt.show()
