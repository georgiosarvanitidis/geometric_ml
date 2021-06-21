import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from sklearn.metrics import pairwise_distances


# Makes a Dx1 vector given x
def my_vector(x):
    return np.asarray(x).reshape(-1, 1)


# Compute pairwise distances between torch matrices (RETURNS |x-y|^2)
def pairwise_dist2_torch(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)
    return dist


# Synthetic datasets
def generate_data(params=None):

    # The semi-circle data in 2D
    if params['data_type'] == 1:
        N = params['N']
        theta = np.pi * np.random.rand(N, 1)
        data = np.concatenate((np.cos(theta), np.sin(theta)), axis=1) + params['sigma'] * np.random.randn(N, 2)
        return data

    # A simple 2-dim surface in 3D with a hole in the middle
    elif params['data_type'] == 2:
        N = params['N']
        Z = np.random.rand(N, 2)
        Z = Z - np.mean(Z, 0).reshape(1, -1)
        Centers = np.zeros((1, 2))
        dists = pairwise_distances(Z, Centers)  # The sqrt(|x|)
        inds_rem = (dists <= params['r']).sum(axis=1)  # N x 1, The points within the ball
        Z_ = Z[inds_rem == 0, :]  # Keep the points OUTSIDE of the ball
        F = (np.sin(2 * np.pi * Z_[:, 0])).reshape(-1, 1)
        F = F + params['sigma'] * np.random.randn(F.shape[0], 1)
        data = np.concatenate((Z_, 0.25 * F), axis=1)
        return data

    # Two moons on a surface and with extra noisy dimensions
    elif params['data_type'] == 3:
        N_all = params['N']
        N = int(N_all / 2)
        theta = np.pi * np.random.rand(N, 1)
        z1 = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
        z2 = np.concatenate((np.cos(theta), -np.sin(theta)), axis=1) + my_vector([1.0, 0.25]).T
        z = np.concatenate((z1, z2), axis=0) + params['sigma'] * np.random.randn(int(N * 2), 2)
        z = z - z.mean(0).reshape(1, -1)
        z3 = (np.sin(np.pi * z[:, 0])).reshape(-1, 1)
        z3 = z3 + params['sigma'] * np.random.randn(z3.shape[0], 1)
        data = np.concatenate((z, 0.5 * z3), axis=1)
        if params['extra_dims'] > 0:
            noise = params['sigma'] * np.random.randn(N_all, params['extra_dims'])
            data = np.concatenate((data, noise), axis=1)
        labels = np.concatenate((0 * np.ones((z1.shape[0], 1)), np.ones((z2.shape[0], 1))), axis=0)
        return data, labels

    return -1


# An implementation of PCA
def my_pca(X, d):
    X_mean = X.mean(axis=0)
    Cov_X = (X - X_mean).T @ (X - X_mean) / X.shape[0]
    eigenValues, eigenVectors = np.linalg.eigh(Cov_X)
    idx = eigenValues.argsort()[::-1]  # Sort the eigenvalues from max -> min
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    V = eigenVectors[:, 0:d]  # The projection matrix  # The projection matrix
    L = eigenValues[0:d]
    mu = X_mean.reshape(-1, 1)  # The center of the dataset
    return V, L, mu


# Plots easily data in 2d or 3d
def my_plot(x, **kwargs):
    if x.shape[1] == 2:
        plt.scatter(x[:, 0], x[:, 1], **kwargs)
        plt.axis('equal')
    if x.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], **kwargs)


# Generate a uniform meshgrid
def my_meshgrid(x1min, x1max, x2min, x2max, N=10):
    X1, X2 = np.meshgrid(np.linspace(x1min, x1max, N), np.linspace(x2min, x2max, N))
    X = np.concatenate((np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)), axis=1)
    return X


# Plots the measure of the Riemannian manifold
def plot_measure(manifold, linspace_x1, linspace_x2, isLog=True, cmapSel=cm.RdBu_r):
    X1, X2 = np.meshgrid(linspace_x1, linspace_x2)
    X = np.concatenate((np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)), axis=1)
    M = manifold.metric_tensor(X.transpose(), nargout=1)

    if manifold.is_diagonal():
        img = np.reshape(np.sqrt(np.prod(M, axis=1)), X1.shape)
    elif not manifold.is_diagonal():
        N = M.shape[0]
        img = np.zeros((N, 1))
        for n in range(N):
            img[n] = np.sqrt(np.linalg.det(np.squeeze(M[n, :, :])))
        img = img.reshape(X1.shape)

    if isLog:
        img = np.log(img + 1e-10)
    else:
        img = img

    plt.imshow(img, interpolation='gaussian', origin='lower',
               extent=(linspace_x1.min(), linspace_x1.max(), linspace_x2.min(), linspace_x2.max()),
               cmap=cmapSel, aspect='equal')


