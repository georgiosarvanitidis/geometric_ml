import numpy as np
from sklearn.metrics import pairwise_distances


# A weighted average of predefined metrics M(x) = \sum w_i(x) M_i, where \sum_i w(x) = 1.
class WeightedAverageMetrics:

    def __init__(self, X, M, sigma, with_projection=False, A=None, b=None):
        self.with_projection = with_projection
        if with_projection:
            self.A = A
            self.b = b.reshape(-1, 1)  # D x 1

        self.X = X  # N x d', the training data (projected if selected)
        self.M = M  # N x 1, the metric tensors on the data
        self.sigma = sigma  # The bandwidth of the kernels

    @staticmethod
    def is_diagonal():
        return False

    def measure(self, z):
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.linalg.det(M)).reshape(-1, 1)  # N x 1

    def metric_tensor(self, c, nargout=1):
        # c: D x N
        # Project the test data
        if self.with_projection:
            c = ((c.T - self.b.T) @ self.A).T  # N x D -> N x d'

        sigma2 = self.sigma ** 2
        D, N = c.shape  # D x N
        cT = c.T  # N x D

        M = np.empty((N, D, D))
        M[:] = np.nan
        if nargout == 2:  # compute the derivative of the metric
            dMdc = np.empty((N, D, D, D))
            dMdc[:] = np.nan

        dist2 = pairwise_distances(cT, self.X) ** 2  # N x N_train
        W = np.exp(-0.5 * dist2 / sigma2)  # The weights w_n(c), N x N_train
        Sum_W = np.sum(W, axis=1, keepdims=True)  # N x 1
        W_norm = W / (Sum_W + 1e-7)  # N x 1
        M = np.einsum('ij,jkl->ikl', W_norm, self.M)

        if nargout == 2:
            WT = W.T  # N_train x N
            dW_n = WT[:, :, None] * (self.X[:, np.newaxis] - cT[np.newaxis, :]) / sigma2  # N_train x N x d
            term1 = dW_n * Sum_W[None, :]  # N_train x N x d
            sum_dW_n = dW_n.sum(axis=0)  # N x d
            term2 = WT[:, :, None] * sum_dW_n[None, :]  # N_train x N x d, same with np.einsum('ij,jk->ijk',WT, sum_dW_n)
            temp = (term1 - term2) / (Sum_W[None, :] ** 2)  # N_train x N x d
            dM = np.einsum('ijk,ilp->jlpk', temp, self.M)

        if nargout == 1:
            return M
        if nargout == 2:
            return M, dM


# This is an implementation of the Euclidean metric
class EuclideanAmbient:

    def __init__(self, D):
        self.D = D

    @staticmethod
    def is_diagonal():
        return True

    @staticmethod
    def measure(z):
        # z: d x N
        return np.ones((z.shape[1], 1))  # N x 1

    def metric_tensor(self, c, nargout=1):
        # c: D x N
        M = np.ones((self.D, c.shape[1])).T  # N x D

        if nargout == 2:
            dM = np.zeros((c.shape[1], c.shape[0], c.shape[0]))
            return M, dM

        return M


# A simple exponential cost function ambient metric with entries m(x) = \sum_i w_i * exp(-0.5*||x - c_i||^2 / sigma2)
class ExponentialRBFCostFunction:

    def __init__(self, centers, W, sigma, rho=1, with_projection=False, A=None, b=None):
        self.with_projection = with_projection
        if with_projection:  # If there is a projection, use it for the centers
            centers = (centers - b.reshape(1, -1)) @ A  # NxD -> N x d'
            self.A = A
            self.b = b.reshape(-1, 1)  # D x 1
        else:
            centers = centers  # NxD

        self.centers = centers  # K x D, the training data
        self.W = W  # K x 1, the output for the data
        self.sigma = sigma
        self.rho = rho  # If the cost does not act, use the Euclidean metric

    @staticmethod
    def is_diagonal():
        return True

    def measure(self, z):
        # z: d x N
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.prod(M, axis=1)).reshape(-1, 1)  # N x 1

    def metric_tensor(self, c, nargout=1):
        # c is D x N

        # c is D x N
        if self.with_projection:
            c = ((c.T - self.b.T) @ self.A).T  # N x D -> N x d', (project the data)

        sigma2 = self.sigma ** 2
        D, N = c.shape
        cT = c.T  # N x D

        M = np.empty((N, D))
        M[:] = np.nan
        if nargout == 2:  # compute the derivative of the metric
            dMdc = np.empty((N, D, D))
            dMdc[:] = np.nan

        dist2 = pairwise_distances(cT, self.centers) ** 2  # N x K
        E = np.exp(-0.5 * dist2 / sigma2)  # The Exponential evaluation, N x K
        F_vals = E @ self.W + self.rho
        M = np.ones((N, D)) * F_vals

        if nargout == 2:
            dif_data_c = (self.centers[:, np.newaxis] - cT[np.newaxis, :])  # K x N x D
            E_W = np.expand_dims(E.T * self.W.repeat(N, 1), axis=2).repeat(D, axis=2) / sigma2
            temp = (E_W * dif_data_c).sum(axis=0)  # N x D
            dMdc = temp[:, np.newaxis, :].repeat(D, axis=1)

        if nargout == 1:
            return M
        elif nargout == 2:
            return M, dMdc


# The pull-back metric when the ambient space is not Euclidean
class ManifoldPullBack:

    def __init__(self, genParams):

        self.with_projection = genParams['with_projection']
        self.A_proj = genParams['A_proj']  # d' x D
        self.b_proj = genParams['b_proj']  # D x 1
        self.d_prime = self.A_proj.shape[0]

        self.manifold_ambient = genParams['manifold_ambient']
        self.beta = genParams['beta']

        self.W2 = genParams['W2']
        self.W1 = genParams['W1']
        self.W0 = genParams['W0']
        self.b2 = genParams['b2']
        self.b1 = genParams['b1']
        self.b0 = genParams['b0']

        self.A_gen = genParams['A_gen']
        self.b_gen = genParams['b_gen']

        self.Wrbf = genParams['Wrbf']  # The weights for the RBFs (D x K)
        self.Crbf = genParams['Crbf']  # The centers for the RBFs (K x d)
        self.Grbf = genParams['Grbf']  # The precision for the RBFs (K x 1)
        self.zeta = genParams['zeta']  # A small value to prevent division by 0

        self.D = self.Wrbf.shape[0]
        self.d = self.Crbf.shape[1]

        self.beta = genParams['beta']

        # Read the hidden and output layer activation functions
        self.act_fun_hidden, self.d_act_fun_hidden, self.dd_act_fun_hidden = \
            activation_of_hidden_layer(genParams)

        self.act_fun_output, self.d_act_fun_output, self.dd_act_fun_output = \
            activation_of_output_layer(genParams)

    @staticmethod
    def is_diagonal():
        return False

    # Compute the measure
    def measure(self, z):
        # z: d x N
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.linalg.det(M)).reshape(-1, 1)  # N x 1

    # Implements the RBF function f(z) = x, output is D x N, input is (d x N)
    def rbf_fun(self, z, nargout=1):
        # z (d x N), the pairwise_distances should take (N x d)
        val = np.exp(-self.Grbf * (pairwise_distances(self.Crbf, z.T) ** 2))
        rbf_val = self.Wrbf @ val + self.zeta
        if nargout == 1:
            return rbf_val
        else:
            return rbf_val, val

    # Implements the derivative of the rbf function
    def d_rbf_fun(self, z):
        # z (d x 1)
        rbf_val = self.rbf_fun(z, nargout=2)[1]  # input should be 1 x d
        return -2 * self.Wrbf @ ((self.Grbf * (z.T - self.Crbf)) * rbf_val)

    def dd_rbf_fun(self, z, d):
        # z (d x 1)
        temp = np.zeros(self.Crbf.shape)
        temp[:, d] = 1

        output_rbf_val, rbf_val = self.rbf_fun(z, nargout=2)  # input should be 1 x d

        val = -0.5 * (((-1.5 / np.sqrt(output_rbf_val ** 5)) * (
                self.Wrbf @ (-2 * self.Grbf * (rbf_val * (z[d] - self.Crbf[:, d].reshape(-1, 1))))))
                      * (-2 * self.Wrbf @ (rbf_val * (self.Grbf * (z.T - self.Crbf))))) \
              + -0.5 * ((1 / np.sqrt(output_rbf_val ** 3))
                        * (self.Wrbf @ (-2 * self.Grbf
                                        * (temp * rbf_val - 2 * (z[d] - self.Crbf[:, d].reshape(-1, 1))
                                           * self.Grbf * (z.T - self.Crbf) * rbf_val))))
        return val

    # Generates a random sample from the generator
    def decode(self, z, random_sample=False, nargout=1):
        # z (d x N)
        mu_val = self.act_fun_output(self.W2 @
                                     self.act_fun_hidden(self.W1 @
                                                         self.act_fun_hidden(
                                                             self.W0 @ z + self.b0) + self.b1) + self.b2)\
                 + self.A_gen @ z + self.b_gen

        if random_sample:
            epsilon_val = np.random.randn(self.D, z.shape[1])
            sigma_val = np.sqrt(1 / self.rbf_fun(z))
            val = mu_val + sigma_val * epsilon_val
            if nargout == 3:
                return val, mu_val, sigma_val
            else:
                return val
        else:
            return mu_val

    # The metric tensor
    def metric_tensor(self, z, nargout=1):
        # z (d x N)
        d, N = z.shape
        M = np.zeros((N, d, d))

        x = self.decode(z)  # d x N -> D x N

        if nargout == 2:
            dMdz = np.zeros((N, d, d, d))
            M_ambient, dM_ambient = self.manifold_ambient.metric_tensor(x, nargout=2)
        else:
            M_ambient = self.manifold_ambient.metric_tensor(x)

        # Pre-compute for speed
        W0Z0b0 = self.W0 @ z + self.b0
        f0W0Z0b0 = self.act_fun_hidden(W0Z0b0)
        W1f0W0Z0b0b1 = self.W1 @ f0W0Z0b0 + self.b1
        W2f1W1f0W0Z0b0b1b2 = self.W2 @ self.act_fun_hidden(W1f0W0Z0b0b1) + self.b2

        for n in range(N):
            z_n = z[:, n].reshape(-1, 1)

            # The ambient metric
            if self.manifold_ambient.is_diagonal():
                M_X_g_z = np.diag(M_ambient[n, :])  # d' x d', the ambient space metric.
            else:
                M_X_g_z = M_ambient[n, :, :]

            # If the linear projection exists
            if self.with_projection:
                M_X_g_z = self.A_proj.T @ M_X_g_z @ self.A_proj

            dmudz = (self.d_act_fun_output(W2f1W1f0W0Z0b0b1b2[:, n].reshape(-1, 1)) * self.W2) \
                    @ (self.d_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1) \
                    @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0) + self.A_gen

            dsigmadz = -0.5 * self.d_rbf_fun(z_n) / np.sqrt(self.rbf_fun(z_n) ** 3)

            M[n, :, :] = dmudz.T @ M_X_g_z @ dmudz + dsigmadz.T @ M_X_g_z @ dsigmadz

            if nargout == 2:

                # Take into account the linear projection if exists
                if self.with_projection:
                    if self.manifold_ambient.is_diagonal():
                        dM_ambient_temp = dM_ambient[n, :, :] @ self.A_proj @ dmudz
                    else:
                        dM_ambient_temp = dM_ambient[n, :, :, :] @ self.A_proj @ dmudz
                else:
                    if self.manifold_ambient.is_diagonal():
                        dM_ambient_temp = dM_ambient[n, :, :] @ dmudz
                    else:
                        dM_ambient_temp = dM_ambient[n, :, :, :] @ dmudz

                for dd in range(d):
                    dJmudzd = (((self.dd_act_fun_output(W2f1W1f0W0Z0b0b1b2[:, n].reshape(-1, 1)) * self.W2)
                                @ (self.d_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1)
                                @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0[:, dd].reshape(-1, 1)))
                               * self.W2) @ ((self.d_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1)
                                             @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0)) \
                              + (self.W2 * self.d_act_fun_output(W2f1W1f0W0Z0b0b1b2[:, n].reshape(-1, 1))) \
                              @ (
                                      (((self.dd_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1)
                                        @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0[:, dd].reshape(
                                                  -1, 1)))
                                       * self.W1) @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0)) \
                              + (self.d_act_fun_output(W2f1W1f0W0Z0b0b1b2[:, n].reshape(-1, 1)) * self.W2) \
                              @ ((self.d_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1)
                                 @ ((self.dd_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1))
                                     * self.W0[:, dd].reshape(-1, 1)) * self.W0))

                    dJsigmadzd = self.dd_rbf_fun(z_n, dd)

                    # Note: separate the cases where the metric is diagonal and if not.
                    if self.manifold_ambient.is_diagonal():
                        dM_ambient_temp_dd = np.diag(dM_ambient_temp[:, dd])
                    else:
                        dM_ambient_temp_dd = dM_ambient_temp[:, :, dd]

                    if self.with_projection:
                        dM_ambient_temp_dd = self.A_proj.T @ dM_ambient_temp_dd @ self.A_proj

                    dMdz[n, :, :, dd] = dJmudzd.T @ M_X_g_z @ dmudz + dmudz.T @ M_X_g_z @ dJmudzd \
                                        + dmudz.T @ dM_ambient_temp_dd @ dmudz \
                                        + dJsigmadzd.T @ M_X_g_z @ dsigmadz + dsigmadz.T @ M_X_g_z @ dJsigmadzd \
                                        + dsigmadz.T @ dM_ambient_temp_dd @ dsigmadz

        if nargout == 2:
            return self.beta * M, self.beta * dMdz
        else:
            return self.beta * M

# Note: local diagonal PCA with projection
# This is the classical local diagonal PCA metric
class LocalDiagPCA:

    def __init__(self, data, sigma, rho, with_projection=False, A=None, b=None):
        self.with_projection = with_projection
        if with_projection:
            self.data = (data - b.reshape(1, -1)) @ A  # NxD
            self.A = A
            self.b = b.reshape(-1, 1)  # D x 1
        else:
            self.data = data  # NxD
        self.sigma = sigma
        self.rho = rho

    @staticmethod
    def is_diagonal():
        return True

    def measure(self, z):
        # z: d x N
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.prod(M, axis=1)).reshape(-1, 1)  # N x 1

    def metric_tensor(self, c, nargout=1):
        # c is D x N
        if self.with_projection:
            c = ((c.T - self.b.T) @ self.A).T

        sigma2 = self.sigma ** 2
        D, N = c.shape

        M = np.empty((N, D))
        M[:] = np.nan
        if nargout == 2:  # compute the derivative of the metric
            dMdc = np.empty((N, D, D))
            dMdc[:] = np.nan

        # TODO: replace the for-loop with tensor operations if possible.
        for n in range(N):
            cn = c[:, n]  # Dx1
            delta = self.data - cn.transpose()  # N x D
            delta2 = delta ** 2  # pointwise square
            dist2 = np.sum(delta2, axis=1, keepdims=True)  # Nx1, ||X-c||^2
            # wn = np.exp(-0.5 * dist2 / sigma2) / ((2 * np.pi * sigma2) ** (D / 2))  # Nx1
            wn = np.exp(-0.5 * dist2 / sigma2)
            s = np.dot(delta2.transpose(), wn) + self.rho  # Dx1
            m = 1 / s  # D x1
            M[n, :] = m.transpose()

            if nargout == 2:
                dsdc = 2 * np.diag(np.squeeze(np.matmul(delta.transpose(), wn)))
                weighted_delta = (wn / sigma2) * delta
                dsdc = dsdc - np.matmul(weighted_delta.transpose(), delta2)
                dMdc[n, :, :] = dsdc.transpose() * m ** 2  # The dMdc[n, D, d] = dMdc_d

        if nargout == 1:
            return M
        elif nargout == 2:
            return M, dMdc


class MlpMeanInvRbfVar:

    def __init__(self, genParams):
        self.W2 = genParams['W2']
        self.W1 = genParams['W1']
        self.W0 = genParams['W0']
        self.b2 = genParams['b2']
        self.b1 = genParams['b1']
        self.b0 = genParams['b0']

        self.Wrbf = genParams['Wrbf']  # The weights for the RBFs (D x K)
        self.Crbf = genParams['Crbf']  # The centers for the RBFs (K x d)
        self.Grbf = genParams['Grbf']  # The precision for the RBFs (K x 1)
        self.zeta = genParams['zeta']  # A small value to prevent division by 0

        self.D = self.Wrbf.shape[0]
        self.d = self.Crbf.shape[1]

        self.beta = genParams['beta']

        # Read the hidden and output layer activation functions
        self.act_fun_hidden, self.d_act_fun_hidden, self.dd_act_fun_hidden = \
            activation_of_hidden_layer(genParams)

        self.act_fun_output, self.d_act_fun_output, self.dd_act_fun_output = \
            activation_of_output_layer(genParams)

    @staticmethod
    def is_diagonal():
        return False

    # Compute the measure
    def measure(self, z):
        # z: d x N
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.linalg.det(M)).reshape(-1, 1)  # N x 1

    # Implements the RBF function f(z) = x, output is D x N, input is (d x N)
    def rbf_fun(self, z, nargout=1):
        # z (d x N), the pairwise_distances should take (N x d)
        val = np.exp(-self.Grbf * (pairwise_distances(self.Crbf, z.T) ** 2))
        rbf_val = self.Wrbf @ val + self.zeta
        if nargout == 1:
            return rbf_val
        else:
            return rbf_val, val

    # Implements the derivative of the rbf function
    def d_rbf_fun(self, z):
        # z (d x 1)
        rbf_val = self.rbf_fun(z, nargout=2)[1]  # input should be 1 x d
        return -2 * self.Wrbf @ ((self.Grbf * (z.T - self.Crbf)) * rbf_val)

    def dd_rbf_fun(self, z, d):
        # z (d x 1)
        temp = np.zeros(self.Crbf.shape)
        temp[:, d] = 1

        output_rbf_val, rbf_val = self.rbf_fun(z, nargout=2)  # input should be 1 x d

        val = -0.5 * (((-1.5 / np.sqrt(output_rbf_val ** 5)) * (
                self.Wrbf @ (-2 * self.Grbf * (rbf_val * (z[d] - self.Crbf[:, d].reshape(-1, 1))))))
                      * (-2 * self.Wrbf @ (rbf_val * (self.Grbf * (z.T - self.Crbf))))) \
              + -0.5 * ((1 / np.sqrt(output_rbf_val ** 3))
                        * (self.Wrbf @ (-2 * self.Grbf
                                        * (temp * rbf_val - 2 * (z[d] - self.Crbf[:, d].reshape(-1, 1))
                                           * self.Grbf * (z.T - self.Crbf) * rbf_val))))
        return val

    # Generates a random sample from the generator
    def decode(self, z, random_sample=False, nargout=1):
        # z (d x N)
        mu_val = self.act_fun_output(self.W2 @
                                     self.act_fun_hidden(self.W1 @
                                                         self.act_fun_hidden(
                                                             self.W0 @ z + self.b0) + self.b1) + self.b2)

        if random_sample:
            epsilon_val = np.random.randn(self.D, z.shape[1])
            sigma_val = np.sqrt(1 / self.rbf_fun(z))
            val = mu_val + sigma_val * epsilon_val
            if nargout == 3:
                return val, mu_val, sigma_val
            else:
                return val
        else:
            return mu_val

    # The metric tensor
    def metric_tensor(self, z, nargout=1):
        # z (d x N)

        d, N = z.shape

        M = np.zeros((N, d, d))

        if nargout == 2:
            dMdz = np.zeros((N, d, d, d))
            # dJ = np.zeros((N, 3, d, d))

        # Pre-compute for speed
        W0Z0b0 = self.W0 @ z + self.b0
        f0W0Z0b0 = self.act_fun_hidden(W0Z0b0)
        W1f0W0Z0b0b1 = self.W1 @ f0W0Z0b0 + self.b1
        W2f1W1f0W0Z0b0b1b2 = self.W2 @ self.act_fun_hidden(W1f0W0Z0b0b1) + self.b2

        for n in range(N):
            z_n = z[:, n].reshape(-1, 1)

            dmudz = (self.d_act_fun_output(W2f1W1f0W0Z0b0b1b2[:, n].reshape(-1, 1)) * self.W2) \
                    @ (self.d_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1) \
                    @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0)

            dsigmadz = -0.5 * self.d_rbf_fun(z_n) / np.sqrt(self.rbf_fun(z_n) ** 3)

            M[n, :, :] = dmudz.T @ dmudz + dsigmadz.T @ dsigmadz

            if nargout == 2:
                for dd in range(d):
                    dJmudzd = (((self.dd_act_fun_output(W2f1W1f0W0Z0b0b1b2[:, n].reshape(-1, 1)) * self.W2)
                                @ (self.d_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1)
                                @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0[:, dd].reshape(-1, 1)))
                               * self.W2) @ ((self.d_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1)
                                             @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0)) \
                              + (self.W2 * self.d_act_fun_output(W2f1W1f0W0Z0b0b1b2[:, n].reshape(-1, 1))) \
                              @ (
                                      (((self.dd_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1)
                                        @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0[:, dd].reshape(
                                                  -1, 1)))
                                       * self.W1) @ (self.d_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1)) * self.W0)) \
                              + (self.d_act_fun_output(W2f1W1f0W0Z0b0b1b2[:, n].reshape(-1, 1)) * self.W2) \
                              @ ((self.d_act_fun_hidden(W1f0W0Z0b0b1[:, n].reshape(-1, 1)) * self.W1)
                                 @ ((self.dd_act_fun_hidden(W0Z0b0[:, n].reshape(-1, 1))
                                     * self.W0[:, dd].reshape(-1, 1)) * self.W0))

                    dJsigmadzd = self.dd_rbf_fun(z_n, dd)

                    dMdz[n, :, :, dd] = dJmudzd.T @ dmudz + dmudz.T @ dJmudzd \
                                        + dJsigmadzd.T @ dsigmadz + dsigmadz.T @ dJsigmadzd

        if nargout == 2:
            return self.beta * M, self.beta * dMdz
        else:
            return self.beta * M


def linear_fun(x):
    return x


def d_linear_fun(x):
    return np.ones(x.shape)


def dd_linear_fun(x):
    return np.zeros(x.shape)


def softplus_fun(x):
    return np.log(1 + np.exp(x))


def d_softplus_fun(x):
    return 1 / (1 + np.exp(-x))


def dd_softplus_fun(x):
    return np.exp(x) / ((1 + np.exp(x)) ** 2)


def tanh_fun(x):
    return np.tanh(x)


def d_tanh_fun(x):
    return 1 - np.tanh(x) ** 2


def dd_tanh_fun(x):
    return -2 * np.tanh(x) * (1 - np.tanh(x) ** 2)


def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid_fun(x):
    return sigmoid_fun(x) * (1 - sigmoid_fun(x))


def dd_sigmoid_fun(x):
    return d_sigmoid_fun(x) * (1 - 2 * sigmoid_fun(x))


############################################################
# Read the hidden layer and output layer activation functions
def activation_of_hidden_layer(genParams):
    if genParams['activation_fun_hidden'] == 'tanh':
        return tanh_fun, d_tanh_fun, dd_tanh_fun
    elif genParams['activation_fun_hidden'] == 'sigmoid':
        return sigmoid_fun, d_sigmoid_fun, dd_sigmoid_fun
    elif genParams['activation_fun_hidden'] == 'softplus':
        return softplus_fun, d_softplus_fun, dd_softplus_fun
    else:
        print('No valid activation function for hidden layer has been specified!')


def activation_of_output_layer(genParams):
    if genParams['activation_fun_output'] == 'tanh':
        return tanh_fun, d_tanh_fun, dd_tanh_fun
    elif genParams['activation_fun_output'] == 'sigmoid':
        return sigmoid_fun, d_sigmoid_fun, dd_sigmoid_fun
    elif genParams['activation_fun_output'] == 'softplus':
        return softplus_fun, d_softplus_fun, dd_softplus_fun
    elif genParams['activation_fun_output'] == 'linear':
        return linear_fun, d_linear_fun, dd_linear_fun
    else:
        print('No valid activation function for output layer has been specified!')
