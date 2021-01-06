import numpy as np
from core import utils
from core import geodesics
from core import geometric_methods
from core import manifolds
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
'''
In this example we fit a LAND model on a simple data manifold
'''

# Generate the data
params = {'N': 200, 'data_type': 1, 'sigma': 0.1}
data = utils.generate_data(params)

# Construct the manifold
manifold = manifolds.LocalDiagPCA(data=data, sigma=0.15, rho=1e-2)

# Plot the Riemannian volume element
utils.plot_measure(manifold, np.linspace(-1.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
utils.my_plot(data, c='k', s=10)
plt.axis('image')
plt.pause(0.1)

# Prepare the solver
solver = geodesics.SolverBVP(NMax=100, tol=1e-1)

# Fit the LAND model
LAND_DATA = KMeans(n_clusters=60).fit(data).cluster_centers_
params = {}
params['means'] = KMeans(n_clusters=2).fit(data).cluster_centers_
params['K'] = 2
params['S'] = 100
params['max_iter'] = 10
params['tol'] = 0.1
params['step_size'] = 0.1
params['mixing_param'] = 0  # [0, 1] how much between empirical covariance and identity
land_res_prior = geometric_methods.land_mixture_model(manifold=manifold,
                                                      solver=solver,
                                                      data=LAND_DATA,
                                                      param=params)

