import numpy as np
from core import geodesics
from core import utils
from core import geometric_methods
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


# Using a trained land model, predict the cluster for the given data
def land_predict(manifold, solver, land, data):
    N, D = data.shape
    K = land['means'].shape[0]
    responsibilities = np.zeros((N, K))
    Logmaps = np.zeros((K, N, D))

    for k in range(K):
        mu = land['means'][k, :].reshape(-1, 1)
        for n in range(N):
            print('[Center: {}/{}] [Point: {}/{}]'.format(k+1, K, n+1, N))
            _, logmap, _, failed, _ \
                = geodesics.compute_geodesic(solver, manifold, mu, data[n, :].reshape(-1, 1))
            Logmaps[k, n, :] = logmap.flatten()  # Use the geodesic even if it is the failed line

    for k in range(K):
        responsibilities[:, k] = land['Weights'][k] \
                             * geometric_methods.evaluate_pdf_land(Logmaps[k, :, :],
                                                                   land['Sigmas'][k, :, :],
                                                                   land['Consts'][k]).flatten()

    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
    return responsibilities


# This function estimates the normalization constant using Monte Carlo sampling on the tangent space
def estimate_norm_constant(manifold, mu, Sigma, S):

    D = Sigma.shape[0]
    Z_eucl = np.sqrt(((2 * np.pi) ** D) * np.linalg.det(Sigma))  # The Euclidean normalization constant

    # Initialize the matrices for the samples and the matrix A
    L, U = np.linalg.eigh(Sigma)
    A = U @ np.diag(np.sqrt(L))
    V_samples = np.zeros((S, D))
    V_samples[:] = np.nan
    X_samples = np.zeros((S, D))
    X_samples[:] = np.nan

    s = 0
    while True:
        try:
            v = A @ np.random.randn(D, 1)  # D x 1, get white noise to sample from N(0, Sigma)
            curve, failed = geodesics.expmap(manifold, x=mu.reshape(-1, 1), v=v.reshape(-1, 1))
            if not failed:
                X_samples[s, :] = curve(1)[0].flatten()
                V_samples[s, :] = v.flatten()
                s = s + 1
        except Exception:
            print('Expmap failed!')

        if s == S:  # We have collected all the samples we need
            break

    inds = np.isnan(X_samples[:, 0])  # The failed exponential maps
    X_samples = X_samples[~inds, :]  # Keep the non-failed ones
    V_samples = V_samples[~inds, :]

    volM_samples = manifold.measure(X_samples.T).flatten()  # Compute the volume element sqrt(det(M))
    norm_constant = np.mean(volM_samples) * Z_eucl  # Estimate the normalization constant

    return norm_constant, V_samples, volM_samples, Z_eucl, X_samples


# Evaluate the pdf for the LAND
def evaluate_pdf_land(Logmaps, Sigma, Const):
    N, D = Logmaps.shape
    inds = np.isnan(Logmaps[:, 0])  # For the logmaps which are NaN return a zero pdf

    result = np.zeros((N, ))
    result[~inds] = (np.exp(-0.5 * pairwise_distances(Logmaps[~inds, :], np.zeros((1, D)),
                                                      metric='mahalanobis',
                                                      VI=np.linalg.inv(Sigma)) ** 2) / Const).flatten()

    result[inds] = 1e-9  # For the failed ones put a very small pdf

    return result.reshape(-1, 1)  # N x 1


# Update one LAND mean using gradient descent
def update_mean(manifold, solver, data, land, k, param):
    N, D = data.shape
    Logmaps = land['Logmaps'][k, :, :]  # N x D
    mu = land['means'][k, :].reshape(-1, 1)  # D x 1
    Sigma = land['Sigmas'][k, :, :]  # D x D
    Resp = land['Resp'][:, k].reshape(-1, 1)  # N x 1
    Const = land['Consts'][k]
    V_samples = land['V_samples'][k, :, :]  # S x D, Use the samples from the last Const estimation
    volM_samples = land['volM_samples'][:, k].reshape(-1, 1)  # S x 1
    Z_eucl = land['Z_eucl'][k]

    for iteration in range(param['max_iter']):

        print('[Updating mean: {}] [Iteration: {}/{}]'.format(k+1, iteration+1, param['max_iter']))

        # Use the non-failed only
        inds = np.isnan(Logmaps[:, 0])  # The failed logmaps
        grad_term1 = Logmaps[~inds, :].T @ Resp[~inds, :] / Resp[~inds, :].sum()
        grad_term2 = V_samples.T @ volM_samples.reshape(-1, 1) * Z_eucl / (Const * param['S'])

        # We do not multiply in front the invSigma because we use the steepest direction
        grad = (grad_term1 - grad_term2)
        curve, failed = geodesics.expmap(manifold, mu, grad * param['step_size'])
        mu_new = curve(1)[0]

        # Compute the logmaps for the new mean
        for n in range(N):
            _, logmap, curve_length, failed, sol \
                = geodesics.compute_geodesic(solver, manifold, mu_new, data[n, :].reshape(-1, 1))
            if failed:
                Logmaps[n, :] = np.nan  # We do NOT use the straight geodesic
            else:
                Logmaps[n, :] = logmap.flatten()

        # Compute the constant for the new mean
        Const, V_samples, volM_samples, Z_eucl, _ = \
            geometric_methods.estimate_norm_constant(manifold, mu_new, Sigma, param['S'])

        cond = np.sum((mu - mu_new) ** 2)  # The condition to stop
        mu = mu_new.copy()
        if cond < param['tol']:   # Stop if lower than tolerance the change
            break

    return mu.flatten(), Logmaps, Const, V_samples, volM_samples.flatten(), Z_eucl


# Update the Sigma using gradient descent TODO: use manifold optimization?
def update_Sigma(manifold, solver, data, land, k, param):
    N, D = data.shape
    Logmaps = land['Logmaps'][k, :, :]  # N x D
    Resp = land['Resp'][:, k].reshape(-1, 1)  # N x 1
    mu = land['means'][k, :].reshape(-1, 1)  # D x 1
    Sigma = land['Sigmas'][k, :, :]  # D x D
    Const = land['Consts'][k]
    V_samples = land['V_samples'][k, :, :]  # S x D
    volM_samples = land['volM_samples'][:, k].reshape(-1, 1)  # S x 1
    Z_eucl = land['Z_eucl'][k]

    # Keep only the non-failed Logmaps
    inds = np.isnan(Logmaps[:, 0])
    grad_term1 = Logmaps[~inds, :].T @ np.diag(Resp[~inds].flatten()) @ Logmaps[~inds, :] / Resp[~inds].sum()  # First term of the gradient
    for iteration in range(param['max_iter']):

        print('[Updating Sigma: {}] [Iteration: {}/{}]'.format(k+1, iteration+1, param['max_iter']))

        # Get the matrix A
        L, U = np.linalg.eigh(np.linalg.inv(Sigma))
        A = np.diag(np.sqrt(L)) @ U.T

        # Compute the gradient
        grad_term2 = (V_samples.T @ np.diag(volM_samples.flatten()) @ V_samples) * Z_eucl / (Const * param['S'])
        grad = A @ (grad_term1 - grad_term2)

        # Update the precision on the tangent space and get the covariance
        A_new = A - grad * param['step_size']
        Sigma_new = np.linalg.inv(A_new.T @ A_new)

        # Compute the new normalization constant
        Const, V_samples, volM_samples, Z_eucl, _ = \
            geometric_methods.estimate_norm_constant(manifold, mu, Sigma_new, param['S'])

        # The stopping condition
        # cond = np.sum((A - A_new) ** 2)
        cond = 0.5 * (np.log(np.linalg.det(Sigma) / np.linalg.det(Sigma_new))
                      + np.trace(np.linalg.inv(Sigma) @ Sigma_new) - D)  # KL-divergence for same mean Gaussians

        Sigma = Sigma_new.copy()
        if cond < param['tol']:
            break

    return Sigma, Const, V_samples, volM_samples.flatten(), Z_eucl


# Compute the negative log likelihood for the given land
def compute_negLogLikelihood(land):
    K = land['means'].shape[0]
    result = np.zeros((land['Logmaps'].shape[1], 1))
    for k in range(K):
        result += land['Weights'][k] * geometric_methods.evaluate_pdf_land(land['Logmaps'][k, :, :],
                                                                           land['Sigmas'][k, :, :],
                                                                           land['Consts'][k])
    return -np.sum(np.log(result))


def land_mixture_model(manifold, solver, data, param):
    K = param['K']

    N, D = data.shape

    # Initializations
    solutions = {}

    # Initialize the land
    land = {}
    land['Resp'] = np.zeros((N, K))  # 1 failed, 0 not failed
    land['Logmaps'] = np.zeros((K, N, D))
    land['Logmaps'][:] = np.nan
    land['Consts'] = np.zeros((K, 1))
    land['means'] = param['means'].copy()
    land['Sigmas'] = np.zeros((K, D, D))
    land['Weights'] = np.zeros((K, 1))  # The mixing components
    land['Failed'] = np.zeros((N, K))  # 1 failed, 0 not failed
    land['V_samples'] = np.zeros((K, param['S'], D))
    land['volM_samples'] = np.zeros((param['S'], K))
    land['Z_eucl'] = np.zeros((K, 1))

    # Initialize components
    for k in range(K):
        for n in range(N):

            print('[Initialize: {}/{}] [Process point: {}/{}]'.format(k+1, K, n+1, N))
            key = 'k_' + str(k) + '_n_' + str(n)
            _, logmap, curve_length, failed, sol \
                = geodesics.compute_geodesic(solver, manifold,
                                             land['means'][k, :].reshape(-1, 1), data[n, :].reshape(-1, 1))
            if failed:
                land['Failed'][n, k] = True
                land['Logmaps'][k, n, :] = logmap.flatten()  # The straight line geodesic
                land['Resp'][n, k] = 1/curve_length  # If points are far lower responsibility
                solutions[key] = None
            else:
                land['Failed'][n, k] = False
                land['Logmaps'][k, n, :] = logmap.flatten()
                land['Resp'][n, k] = 1/curve_length
                solutions[key] = sol

    land['Resp'] = land['Resp'] / land['Resp'].sum(axis=1, keepdims=True)   # Compute the responsibilities
    land['Weights'] = np.sum(land['Resp'], axis=0).reshape(-1, 1) / N

    # Use the closest points to estimate the Sigmas and the normalization constants
    for k in range(K):
        inds_k = (land['Resp'].argmax(axis=1) == k)  # The closest points to the k-th center
        land['Sigmas'][k, :, :] = np.cov(land['Logmaps'][k, inds_k, :].T) * (1 - param['mixing_param']) + np.eye(D) * param['mixing_param']
        land['Consts'][k], land['V_samples'][k, :, :], land['volM_samples'][:, k], land['Z_eucl'][k], _ \
            = geometric_methods.estimate_norm_constant(manifold, land['means'][k, :].reshape(-1, 1), land['Sigmas'][k, :, :], param['S'])

    negLogLikelihood = geometric_methods.compute_negLogLikelihood(land)
    negLogLikelihoods = [negLogLikelihood]

    for iteration in range(param['max_iter']):
        print('[Iteration: {}/{}] [Negative log-likelihood: {}]'.format(iteration+1, param['max_iter'], negLogLikelihood))

        # ----- E-step ----- #
        for k in range(K):
            land['Resp'][:, k] = land['Weights'][k] \
                                 * geometric_methods.evaluate_pdf_land(land['Logmaps'][k, :, :],
                                                                       land['Sigmas'][k, :, :],
                                                                       land['Consts'][k]).flatten()
        land['Resp'] = land['Resp'] / land['Resp'].sum(axis=1, keepdims=True)

        # ----- M-step ----- #
        # Update the means
        for k in range(K):
            land['means'][k, :], land['Logmaps'][k, :, :], land['Consts'][k], land['V_samples'][k, :, :], land['volM_samples'][:, k], land['Z_eucl'][k] = \
                geometric_methods.update_mean(manifold, solver, data, land, k, param)
        # Update the covariances
        for k in range(K):
            land['Sigmas'][k, :, :], land['Consts'][k], land['V_samples'][k, :, :], land['volM_samples'][:, k], land['Z_eucl'][k] = \
                geometric_methods.update_Sigma(manifold, solver, data, land, k, param)
        # Update the constants
        for k in range(K):
            land['Consts'][k], land['V_samples'][k, :, :], land['volM_samples'][:, k], land['Z_eucl'][k], _ \
                = geometric_methods.estimate_norm_constant(manifold, land['means'][k, :].reshape(-1, 1), land['Sigmas'][k, :, :], param['S'])
        # Update the mixing components
        land['Weights'] = np.sum(land['Resp'], axis=0).reshape(-1, 1) / N

        # Compute the new likelihood and store it
        newNegLogLikelihood = geometric_methods.compute_negLogLikelihood(land)
        negLogLikelihoods = np.concatenate((negLogLikelihoods, [newNegLogLikelihood]), axis=0)

        # Check the difference in log-likelihood between updates
        if (newNegLogLikelihood - negLogLikelihood) ** 2 < param['tol']:
            break
        else:
            negLogLikelihood = newNegLogLikelihood

    land['negLogLikelihoods'] = negLogLikelihoods
    return land













