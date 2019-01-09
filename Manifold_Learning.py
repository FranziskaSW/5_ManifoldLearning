import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D


def digits_example():
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()

def swiss_roll_example():
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
    Example code to show you how to load the faces data.
    '''
    path = 'faces.pickle'
    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels**0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
    A plot function for viewing images in their embedded locations. The
    function receives the embedding (X) and the original images (images) and
    plots the images along with the embeddings.

    :param X: Nxd embedding matrix (after dimensionality reduction).
    :param images: NxD original data matrix of images.
    :param title: The title of the plot.
    :param num_to_plot: Number of images to plot along with the scatter plot.
    :return: the figure object.
    '''

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig

def squared_euclid(X, Y):
    """
    return the pair-wise squared euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """

    dist = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(0, X.shape[0]):
        for j in range(0, Y.shape[0]):
            dist[i,j] = (np.sqrt(np.dot(X[i], X[i].T) - 2 * np.dot(X[i], Y[j].T) + np.dot(Y[j], Y[j].T)))**2
    return dist


def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    data, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)  # TODO: delete
    X = squared_euclid(data, data)
    d = 2

    n = X.shape[0]
    H = np.eye(n) - 1/n * np.ones((n, n))
    S = -1/2 * np.dot(np.dot(H, X), H)
    v, U = np.linalg.eigh(S)            # TODO: does not work... or maybe it does but just not on this dataset
    biggest_v = (-v).argsort()[:d]
    ds_v = v[biggest_v]
    ds_U = U[:,biggest_v]
    ds = np.multiply(np.matrix(np.sqrt(ds_v)), ds_U)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ds[:, 0], ds[:, 1], c=color) #, X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()

    pass

def knn(X, k):
    """
    calculate the nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """

    dist_idx = np.argsort(X, axis=1)
    nearest_idx = dist_idx[:, :(k+1)]

    NN = np.zeros(X.shape)
    for i in range(0, X.shape[0]):
        NN[i, nearest_idx[i]] += 1
    NN = NN  - np.eye(X.shape[0])

    return NN

def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''

    # TODO: YOUR CODE HERE
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)  # TODO: delete
    Distance = squared_euclid(X, X)

    k = 30
    d = 2
    KNN = knn(Distance, k)

    n = KNN.shape[0]
    index = np.arange(0, n)
    W = np.zeros((n, n))

    for i in range(0,n):

        idx_i = index[KNN[i] == 1]
        Z_i = X[idx_i] - X[i]
        G = Z_i.dot(Z_i.T)   # (k x k)
        W_i = np.linalg.pinv(G).dot(np.ones(k))
        W_i = W_i/sum(W_i)
        W[i, idx_i] = W_i.tolist()

    # find Y
    M = np.eye(n) - W
    MTM = (M.T).dot(M)

    v, U = np.linalg.eigh(MTM)
    smallest_v = v.argsort()[1:(d+1)]
    U = U[:,smallest_v]

    ds = U
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ds[:, 0], ds[:, 1], c=color) #, X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()

    # TODO: needs parameter tuning
    pass


def DiffusionMap(X, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    '''

    X, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)  # TODO: delete
    d = 2

    Distance = squared_euclid(X, X)

    sigma = 2

    K = np.exp(-(Distance)/(2*(sigma**2)))  # similarity Matrix
    A = np.linalg.inv(np.diag(K.sum(axis=1))).dot(K)
    v, U = np.linalg.eigh(A)
    biggest_v = (-v).argsort()[1:(d+1)]  # why not the biggest one? usually we only leave out the smallest because the EV is 0
    U = U[:,biggest_v]

    ds = np.multiply(U, v[biggest_v])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ds[:, 0], ds[:, 1], c=color) #, X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()




    pass


def main():
    # here everything

    # TODO: YOUR CODE HERE

if __name__ == '__main__':
    main()

    pass
