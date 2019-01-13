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

    return X, color


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


def MDS(D, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.

    :param D: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''

    n = D.shape[0]
    H = np.eye(n) - 1/n * np.ones((n, n))
    S = -1/2 * np.dot(np.dot(H, D), H)
    v, U = np.linalg.eigh(S)  # eigenvalues sorted from smallest to biggest
    biggest_v = (-v).argsort()[:d]

    ds_v = v[biggest_v]
    ds_U = U[:,biggest_v]
    ds = np.multiply(np.matrix(np.sqrt(ds_v)), ds_U)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(ds[:, 0], ds[:, 1])# , c=color) #, X[:, 2], c=color, cmap=plt.cm.Spectral)
    # plt.show()

    return np.array(ds)


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


def LLE(X, D, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.

    :param X: NxD data matrix.
    :param D: pairwise distance matrix (euclidean square)
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''

    KNN = knn(D, k)

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

    # ds = U
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(ds[:, 0], ds[:, 1])# , c=color) #, X[:, 2], c=color, cmap=plt.cm.Spectral)
    # plt.show()

    return np.array(U)


def DiffusionMap(X, D, d, sigma, t):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    kernel matrix to only the k nearest neighbor of each data point.

    :param X: NxD data matrix.
    :param D: NxN distance matrix, squared euclidean
    :param d: the dimension.
    :param sigma: the sigma of the gaussian for the kernel matrix transformation.
    :param t: the scale of the diffusion (amount of time steps).
    :return: Nxd reduced data matrix.
    '''

    K = np.exp(-(D)/(2*(sigma**2)))  # similarity Matrix
    A = np.linalg.inv(np.diag(K.sum(axis=1))).dot(K)
    v, U = np.linalg.eigh(A)
    # normalize to biggest eigenvalue bzw. biggest eigenvector
    # v = v/v[v.argmax()]
    # U = U/U[:, v.argmax()]
    biggest_v = (-v).argsort()[1:(d+1)]
    U = U[:,biggest_v]

    ds = np.multiply(U, np.power(v[biggest_v], t))

    return np.array(ds)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ds[:, 0], ds[:, 1], ds[:, 2]) # , c=color) #, X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def plane(points):

    x = np.random.uniform(1, 10, points)
    y = np.random.uniform(1, 10, points)
    z = np.zeros(points)
    D = np.matrix([x, y, z]).T

    G = np.random.normal(0, 1, 9).reshape(3,3)
    Q, R = np.linalg.qr(G)

    X = np.array(Q.dot(D.T).T)

    return X


def load_data(name, points=500):

    if name == 'swiss_roll':
        X, labels = datasets.samples_generator.make_swiss_roll(n_samples=points)
    elif name == 'digits':
        digits = datasets.load_digits()
        X = digits.data / 255.
        labels = digits.target
    elif name == 'faces':
        path = 'faces.pickle'
        with open(path, 'rb') as f:
            X = pickle.load(f)
        labels = [0]*X.shape[0]
    elif name == 'plane':
        X = plane(points)
        labels = squared_euclid(np.array([[-1000, -1000, -1000]]), X).flatten()

    return X, labels

# TODO: parameter tuning
# TODO:_lle k
# TODO: dm sigma, t


def plot_3methods(X_mds, X_lle, X_dm, labels):

    results = [X_mds, X_lle, X_dm]

    # plot points2cluster for different k
    fig1, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip([ax0, ax1, ax2], results):
        X_method = k
        ax.scatter(X_method[:, 0], X_method[:, 1], c=labels, cmap=plt.cm.Spectral)

    ax0.set_title('MDS')
    ax1.set_title('LLE')
    ax2.set_title('Diffusion Map')

    return fig1


def plot_with_images_s(X, images, ax, image_num=30):

    n, pixels = np.shape(images)
    img_size = int(pixels**0.5)

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

    return ax


def plot_3methods_faces(data, X_mds, X_lle, X_dm, labels):

    results = [X_mds, X_lle, X_dm]

    # plot points2cluster for different k
    fig1, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip([ax0, ax1, ax2], results):
        X_method = k
        ax = plot_with_images_s(X_method, data, ax)

    ax0.set_title('MDS')
    ax1.set_title('LLE')
    ax2.set_title('Diffusion Map')

    return fig1

def noisy_mds(X, var):

    results = dict()

    for i in range(0, len(var)):
        variance = var[i]

        noise = np.random.normal(0, variance, X.shape[0]*X.shape[1]).reshape((X.shape))
        X_noisy = X + noise

        D = squared_euclid(X_noisy, X_noisy)
        n = D.shape[0]
        H = np.eye(n) - 1/n * np.ones((n, n))
        S = -1/2 * np.dot(np.dot(H, D), H)
        v, U = np.linalg.eigh(S)  # eigenvalues sorted from smallest to biggest
        v[::-1].sort()

        v_10 = v[:10]
        results.update({variance: v[:10]})

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,12))

    for ax, k in zip(axes.flat, sorted(results)):
        print(k)
        vals = results[k]
        ax.bar(range(0, 10), vals, color=np.array(['b', 'b', 'r']))
        ax.set_title('$\sigma_{noise}$ = ' + str(k))

    return fig


def tune_lle(X, D, labels, neighbors):

    results = dict()

    for i in range(0, len(neighbors)):
        k = neighbors[i]
        # X_lle = LLE(X, D, 2, k)
        X_lle = DiffusionMap(X, D, 2, k, 2)
        results.update({k: X_lle})

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,12),
                             subplot_kw={'xticks': [], 'yticks': []})

    for ax, k in zip(axes.flat, sorted(results)):
        print(k)
        X = results[k]
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Spectral)
        ax.set_title('k = ' + str(k))

    return fig


def plot_scree():

    return fig


def plot_distance_scatter():

    return fig


def main():

    X, labels = load_data('swiss_roll', points=1000)

    D = squared_euclid(X, X)

    X_lle = LLE(X, D, 2, 20)
    X_dm = DiffusionMap(X, D, 2, 12, 1)
    X_mds = MDS(D, 2)

    # var = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 50]
    # fig_1 = noisy_mds(X, var)

    neighbors = [5, 10, 15, 20, 25, 30, 35, 50, 100]
    neighbors = [3, 5, 6, 7, 8, 10, 11, 12, 13]
    var = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 50]
    var = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]# swiss_roll

    # fig_2 = tune_lle(X, D, labels, neighbors)
    fig_3 = tune_lle(X, D, labels, var)

    fig = plot_3methods(X_mds, X_lle, X_dm, labels)
    #fig = plot_3methods_faces(X, X_mds, X_lle, X_dm, labels)
    plt.show()


if __name__ == '__main__':
    main()

    pass
#
# #X, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)  # TODO: delete
# X = plane(500)
# d = 2
#
#
# Distance = squared_euclid(X, X)
#
# sigma = 12
# t = 1
#
# K = np.exp(-(Distance)/(2*(sigma**2)))  # similarity Matrix
# A = np.linalg.inv(np.diag(K.sum(axis=1))).dot(K)
#
# v, U = np.linalg.eigh(A)
# v = v/v[v.argmax()]
# U = U/U[:, v.argmax()]
# biggest_v = (-v).argsort()[1:(d+1)]
# U = U[:,biggest_v]
#
# ds = np.multiply(U, np.power(v[biggest_v], t))
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(ds[:, 0], ds[:, 1]) #, c=color) #, X[:, 2], c=color, cmap=plt.cm.Spectral)
# plt.show()
#
#
#
# X, color = swiss_roll_example()
# # Values of epsilon in base 2 we want to scan.
# sig =  np.power(2, np.arange(-10.,14.,1))
# sig = np.linspace(0.1, 20)
#
# # Pre-allocate array containing sum(Aij).
# Aij = np.zeros(sig.shape)
#
# from sklearn import metrics
#
# # Loop through values of epsilon and evaluate matrix sum.
# for i in range(len(sig)):
#     A = metrics.pairwise.rbf_kernel(X, gamma=1./(2.*sig[i]**2))
#     Aij[i] = A.sum()
#
# plt.semilogy(range(len(sig)), Aij)