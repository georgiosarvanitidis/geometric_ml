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


# Plots easily data in 2d or 3d
def my_plot(x, **kwargs):
    if x.shape[1] == 2:
        plt.scatter(x[:, 0], x[:, 1], **kwargs)
        plt.axis('equal')
    if x.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], **kwargs)


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


