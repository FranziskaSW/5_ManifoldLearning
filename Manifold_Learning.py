import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D


def plane(points):
    '''
    creates a 2d plane in a 3d space

    :param points: size of dataset
    :return: (points x 3) dimensional dataset
    '''
    x = np.random.uniform(1, 10, points)
    y = np.random.uniform(1, 10, points)
    z = np.zeros(points)
    D = np.matrix([x, y, z]).T

    G = np.random.normal(0, 1, 9).reshape(3, 3)
    Q, R = np.linalg.qr(G)

    X = np.array(Q.dot(D.T).T)

    return X


def load_data(name, points=500):
    '''
    loads the dataset specified by name

    :param name: name of dataset
    :param points: size of dataset
    :return: dataset and its labels
    '''
    if name == 'swiss_roll':
        X, labels = datasets.samples_generator.make_swiss_roll(n_samples=points)

    elif name == 'mnist':
        mnist = datasets.load_digits()
        X = mnist.data / 255.
        labels = mnist.target

    elif name == 'faces':
        path = 'faces.pickle'
        with open(path, 'rb') as f:
            X = pickle.load(f)
        labels = [0] * X.shape[0]

    elif name == 'plane':
        X = plane(points)
        labels = squared_euclid(np.array([[-1000, -1000, -1000]]), X).flatten()

    return X, labels


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
    biggest_v = (-v).argsort()[1:(d+1)]
    U = U[:,biggest_v]

    ds = np.multiply(U, np.power(v[biggest_v], t))

    return np.array(ds)


#######################################################################################
#                     Parameter Tuning                                                #
#######################################################################################

def tune_lle(X, D, labels, neighbors):
    '''
    runs LLE for different neighbors-value. Plots the result

    :param X: NxD data matrix.
    :param D: pairwise distance matrix (euclidean square)
    :param labels: labels of data
    :param neighbors: list of neighbor-values (max. 9)
    :return: plot of the results
    '''

    results = dict()

    for i in range(0, len(neighbors)):
        k = neighbors[i]
        X_lle = LLE(X, D, 2, k)
        results.update({k: X_lle})

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,12),
                             subplot_kw={'xticks': [], 'yticks': []})

    for ax, k in zip(axes.flat, sorted(results)):
        print(k)
        X = results[k]
        ax.scatter(X[:, 0], X[:, 1],  marker='.', alpha=0.7, c=labels, cmap=plt.cm.Spectral)
        ax.set_title('k = ' + str(k))

    return fig


def tune_dm(X, D, labels, sigma):
    '''
    runs DiffusionMap for different sigma values of heat kernel. Plots the results

    :param X: NxD data matrix.
    :param D: pairwise distance matrix (euclidean square)
    :param labels: labels of data
    :param sigma: list of sigma-values (max. 9)
    :return: plot of the results
    '''

    results = dict()

    for k in range(0, len(sigma)):
        sig = sigma[k]
        X_dm = DiffusionMap(X, D, 2, sig, 1)
        results.update({sig: X_dm})

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,12),
                             subplot_kw={'xticks': [], 'yticks': []})

    for ax, k in zip(axes.flat, sorted(results)):
        print('sigma:' + str(k))
        X = results[k]
        ax.scatter(X[:, 0], X[:, 1],  marker='.', alpha=0.7, c=labels, cmap=plt.cm.Spectral)
        ax.set_title('sigma = ' + str(k))

    return fig


def tune_dm_t(X, D, labels, sigma, times):
    '''
    runs DiffusionMap for different time values for diffusion. Plots the results

    :param X: NxD data matrix.
    :param D: pairwise distance matrix (euclidean square)
    :param labels: labels of data
    :param sigma: fix sigma value
    :param times: list of sigma-values (max. 9)
    :return: plot of the results
    '''

    results = dict()

    for t in range(0, len(times)):
        time = times[t]
        print(t, time)
        X_dm = DiffusionMap(X, D, 2, sigma, time)
        results.update({time: X_dm})

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,12),
                             subplot_kw={'xticks': [], 'yticks': []})

    for ax, k in zip(axes.flat, sorted(results)):
        print('time:' + str(k))
        X = results[k]
        ax.scatter(X[:, 0], X[:, 1],  marker='.', alpha=0.7, c=labels, cmap=plt.cm.Spectral)
        ax.set_title('t = ' + str(k))

    return fig


#######################################################################################
#                     Plotting and Evaluation                                         #
#######################################################################################

def plot_dataset(X, labels):
    '''
    creates 3d scatter plot of Data
    :param X: Nx3 Data matrix
    :param labels: labels of data
    :return: 3d catter plot
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap=plt.cm.Spectral)

    return ax


def plot_3methods(X_mds, X_lle, X_dm, labels):
    '''
    creates 2d scatter plots of result of three methods
    :param X_mds: Nx2 result matrix from MDS
    :param X_lle: x2 result matrix from LLE
    :param X_dm: x2 result matrix from Diffusion Map
    :param labels: labels of data
    :return: plot
    '''

    results = [X_mds, X_lle, X_dm]
    fig1, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip([ax0, ax1, ax2], results):
        X_method = k
        ax.scatter(X_method[:, 0], X_method[:, 1],  marker='.', alpha=0.7, c=labels, cmap=plt.cm.Spectral)

    ax0.set_title('MDS')
    ax1.set_title('LLE')
    ax2.set_title('Diffusion Map')

    return fig1


def plot_with_images_s(X, images, ax, image_num=30):
    '''
    subplot for 3methods_faces - creates scatter plot of faces data with example images
    :param X: Nx2 data matrix
    :param images: original faces data
    :param ax: subplot
    :param image_num: how many images should be displayed in plot
    :return: subplot
    '''

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
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7, cmap=plt.cm.Spectral)

    return ax


def plot_3methods_faces(data, X_mds, X_lle, X_dm, labels):
    '''
    creates 2d scatter plots of result of three methods with example images
    :param X_mds: Nx2 result matrix from MDS
    :param X_lle: x2 result matrix from LLE
    :param X_dm: x2 result matrix from Diffusion Map
    :param labels: labels of data
    :return: plot
    '''

    results = [X_mds, X_lle, X_dm]
    fig1, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4),
                             subplot_kw={'xticks': [], 'yticks': []})

    fig1.subplots_adjust(hspace=0.3, wspace=0.05)

    for ax, k in zip([ax0, ax1, ax2], results):
        X_method = k
        ax = plot_with_images_s(X_method, data, ax, 30)

    ax0.set_title('MDS')
    ax1.set_title('LLE')
    ax2.set_title('Diffusion Map')

    return fig1


def noisy_mds(X, var):
    '''
    creates scree plots of eigenvalues from MDS. The input dataset for MDS varies through
    different random noise values which are characterized by its variance var.
    :param X: NxD data set
    :param var: list of max. 9 variance values for random noise
    :return: scree plot
    '''

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
        results.update({variance: v[:3]})

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,12))

    for ax, k in zip(axes.flat, sorted(results)):
        print(k)
        vals = results[k]
        ax.bar(range(0, 3), vals, color=np.array(['b', 'b', 'r']))
        ax.set_title('$\sigma_{noise}$ = ' + str(k))

    return fig


def load_parameters(dataset):
    '''
    loads data and its parameters for tuning
    :param dataset: name of dataset (str)
    :return: X: NxD data matrix
             labels: labels of data
             D: pairwise distance matrix (euclidean square)
             neighbors: list of max. 9 entries, k nearest neighbors for LLE tuning
             sigma: list of max. 9 entries, sigma-values for Diffusion Map tuning
             times: list of max. 9 entries, time-values for Diffusion Map tuning
    '''

    if dataset == 'swiss_roll':
        X, labels = load_data(dataset , points=2000)
        D = squared_euclid(X, X)
        neighbors = [5, 10, 15, 20, 25, 30, 35, 50, 100]
        sigma = [1.15, 1.3, 1.45, 1.6, 1.75, 1.9, 2.05, 2.2, 5]
        times = [1, 2, 3, 4, 10, 15, 20, 25, 100]

    elif dataset == 'mnist':
        X, labels = load_data(dataset)
        print(X.shape, labels.shape)
        D = squared_euclid(X, X)
        neighbors = [2, 5, 7, 9, 10, 11, 12, 15, 20]
        sigma = [0.03, 0.04, 0.042, 0.044, 0.046, 0.048, 0.05, 0.06, 0.08]
        times = [1, 2, 3, 4, 10, 15, 20, 25, 100]

    elif dataset == 'faces':
        X, labels = load_data(dataset)
        D = squared_euclid(X, X)
        neighbors = [2, 4, 10, 14, 15, 16, 17, 20, 30]
        sigma = [3, 5, 6, 7, 9, 11, 15, 25, 50]
        times = [1, 2, 3, 4, 10, 15, 20, 25, 100]

    else:
        print('dataset unknown')
        X, labels, D, neighbors, sigma, times = None, None, None, None, None

    return X, labels, D, neighbors, sigma, times


def main():

    dataset = 'faces'
    X, labels, D, neighbors, sigma, times = load_parameters(dataset)

    # ----------------------------------------------------------------------------------
    # parameter tuning
    # ----------------------------------------------------------------------------------
    fig_2 = tune_lle(X, D, labels, neighbors)
    fig_3 = tune_dm(X, D, labels, sigma)
    fig_4 = tune_dm_t(X, D, labels, 0.044, times)

    fig_2.savefig(str(dataset) + '_LLETuning.png')
    fig_3.savefig(str(dataset) + '_DMTuning.png')
    fig_4.savefig(str(dataset) + '_DMTuning_t.png')

    # ----------------------------------------------------------------------------------
    # plot comparison of tuned algorithms
    # ----------------------------------------------------------------------------------
    X_mds = MDS(D, 2)
    X_lle = LLE(X, D, 2, 15) # k=15 for swiss_roll 2000 # 15 faces # 10 mnist
    X_dm = DiffusionMap(X, D, 2, 7, 1) # 1.75 swiss_roll # 7 faces # 0.044 mnist

    # fig_5 = plot_3methods(X_mds, X_lle, X_dm, labels)
    fig_5 = plot_3methods_faces(X, X_mds, X_lle, X_dm, labels)
    fig_5.savefig(str(dataset) + '_comparison.png')

    # ----------------------------------------------------------------------------------
    # scree plot for MDS algorithm with noisy plane in 3d
    # ----------------------------------------------------------------------------------
    X, labels = load_data('plane', 2000)
    var = [0.1, 0.5, 1, 2, 3, 4, 5, 10, 20]
    fig_6 = noisy_mds(X, var)
    fig_6.savefig('screeplot.png')

    plt.show()


if __name__ == '__main__':
    main()

    pass
