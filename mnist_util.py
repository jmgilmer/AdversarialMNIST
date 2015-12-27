import matplotlib.pyplot as plt
import numpy as np
def plot_mnist_digit(image):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()