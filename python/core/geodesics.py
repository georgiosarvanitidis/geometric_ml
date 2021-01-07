import numpy as np
import sys
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import CubicSpline


# This function evaluates the differential equation c'' = f(c, c')
def geodesic_system(manifold, c, dc):
    # Input: c, dc ( D x N )

    D, N = c.shape
    if (dc.shape[0] != D) | (dc.shape[1] != N):
        print('geodesic_system: second and third input arguments must have same dimensionality\n')
        sys.exit(1)

    # Evaluate the metric and the derivative
    M, dM = manifold.metric_tensor(c, nargout=2)

    # Prepare the output (D x N)
    ddc = np.zeros((D, N))

    # Diagonal Metric Case, M (N x D), dMdc_d (N x D x d=1,...,D) d-th column derivative with respect to c_d
    if manifold.is_diagonal():
        for n in range(N):
            dMn = np.squeeze(dM[n, :, :])
            ddc[:, n] = -0.5 * (2 * np.matmul(dMn * dc[:, n].reshape(-1, 1), dc[:, n])
                                - np.matmul(dMn.T, (dc[:, n] ** 2))) / M[n, :]

    # Non-Diagonal Metric Case, M ( N x D x D ), dMdc_d (N x D x D x d=1,...,D)
    else:

        for n in range(N):
            Mn = np.squeeze(M[n, :, :])
            if np.linalg.cond(Mn) < 1e-15:
                print('Ill-condition metric!\n')
                sys.exit(1)

            dvecMdcn = dM[n, :, :, :].reshape(D * D, D, order='F')
            blck = np.kron(np.eye(D), dc[:, n])

            ddc[:, n] = -0.5 * (np.linalg.inv(Mn) @ (
                    2 * blck @ dvecMdcn @ dc[:, n]
                    - dvecMdcn.T @ np.kron(dc[:, n], dc[:, n])))

    return ddc


# This function changes the 2nd order ODE to two 1st order ODEs takes c, dc and returns dc, ddc.
def second2first_order(manifold, state):
    # Input: state [c; dc] (2D x N), y=[dc; ddc]: (2D x N)
    D = int(state.shape[0] / 2)

    # TODO: Something better for this?
    if state.ndim == 1:
        state = state.reshape(-1, 1)  # (2D,) -> (2D, 1)

    c = state[:D, :]  # D x N
    cm = state[D:, :]  # D x N
    cmm = geodesic_system(manifold, c, cm)  # D x N
    y = np.concatenate((cm, cmm), axis=0)
    return y


# If the solver failed provide the linear distance as the solution
def evaluate_failed_solution(p0, p1, t):
    # Input: p0, p1 (D x 1), t (T x 0)
    c = (1 - t) * p0 + t * p1  # D x T
    dc = np.repeat(p1 - p0, np.size(t), 1)  # D x T
    return c, dc


def evaluate_solution(solution, t):
    # Input: t (Tx0),
    c_dc = solution.sol(t)
    D = int(c_dc.shape[0] / 2)

    if np.size(t) == 1:
        c = c_dc[:D].reshape(D, 1)
        dc = c_dc[D:].reshape(D, 1)
    else:
        c = c_dc[:D, :]  # D x T
        dc = c_dc[D:, :]  # D x T
    return c, dc


def evaluate_spline_solution(curve, dcurve, t):
    # Input: t (Tx0), t_scale is used from the Expmap to scale the curve in order to have correct length,
    #        solution is an object that solver_bvp() returns
    c = curve(t)
    dc = dcurve(t)
    D = int(c.shape[0])

    # TODO: Why the t_scale is used ONLY for the derivative component?
    if np.size(t) == 1:
        c = c.reshape(D, 1)
        dc = dc.reshape(D, 1)
    else:
        c = c.T  # Because the c([0,..,1]) -> N x D
        dc = dc.T
    return c, dc


# This function computes the infinitesimal small length on a curve
def local_length(manifold, curve, t):
    # Input: curve function of t returns (D X T), t (T x 0)
    c, dc = curve(t)  # [D x T, D x T]
    D = c.shape[0]
    M = manifold.metric_tensor(c, nargout=1)
    if manifold.is_diagonal():
        dist = np.sqrt(np.sum(M.transpose() * (dc ** 2), axis=0))  # T x 1, c'(t) M(c(t)) c'(t)
    else:
        dc = dc.T  # D x N -> N x D
        dc_rep = np.repeat(dc[:, :, np.newaxis], D, axis=2)  # N x D -> N x D x D
        Mdc = np.sum(M * dc_rep, axis=1)  # N x D
        dist = np.sqrt(np.sum(Mdc * dc, axis=1))  # N x 1
    return dist


# This function computes the length of the geodesic curve
# The smaller the approximation error (tol) the slower the computation.
def curve_length(manifold, curve, a=0, b=1, tol=1e-5, limit=50):
    # Input: curve a function of t returns (D x ?), [a,b] integration interval, tol error of the integration
    if callable(curve):
        # function returns: curve_length_eval = (integral_value, some_error)
        curve_length_eval = integrate.quad(lambda t: local_length(manifold, curve, t), a, b, epsabs=tol, limit=limit)  # , number of subintervals
    else:
        print("TODO: Not implemented yet integration for discrete curve!\n")
        sys.exit(1)

    return curve_length_eval[0]


# This function plots a curve that is given as a parametric function, curve: t -> (D x len(t)).
def plot_curve(curve, **kwargs):
    N = 1000
    T = np.linspace(0, 1, N)
    curve_eval = curve(T)[0]

    D = curve_eval.shape[0]  # Dimensionality of the curve

    if D == 2:
        plt.plot(curve_eval[0, :], curve_eval[1, :], **kwargs)
    elif D == 3:
        plt.plot(curve_eval[0, :], curve_eval[1, :], curve_eval[2, :], **kwargs)


# This function returns the boundary conditions
def boundary_conditions(ya, yb, c0, c1):
    D = c0.shape[0]
    retVal = np.zeros(2 * D)
    retVal[:D] = ya[:D] - c0.flatten()
    retVal[D:] = yb[:D] - c1.flatten()
    return retVal


# Reparametrizes a curve with arc-length using the Euclidean metric
def unit_speed_curve_euclidean(curve, t, N_points=1000):
    # First make the curve repametrization and then evaluate it at the points t.
    T = np.linspace(0, 1, N_points)
    # N = N_points - 2  # The points without the boundaries
    T_without_boundary_points = T[1:-1]  # Except the boundary values
    local_lengths_N = np.sqrt(np.sum(curve(T_without_boundary_points)[1] ** 2, axis=0, keepdims=True)).T
    lenght_cumsums = np.cumsum(local_lengths_N)  # Local lengths
    total_length = lenght_cumsums[-1] + 0.1
    # Temp is the length on the curve in the interval [0,1], i.e. move 0.1 on the length.
    temp = np.concatenate(([[0]], lenght_cumsums.reshape(-1, 1), [[total_length]]), axis=0).flatten() / total_length
    new_time = CubicSpline(temp, T)  # Here we give the proportion of length to move on the curve, and we get the time
    return curve(new_time(t))


# Reparametrizes a curve with unit speed under the metric of the manifold
def unit_speed_curve(manifold, curve, t, N_points=1000):
    # First make the curve repametrization and then evaluate it at the points t.
    T = np.linspace(0, 1, N_points)
    N = N_points - 2  # The points without the boundaries
    T_without_boundary_points = T[1:-1]  # Except the boundary values
    local_lengths_N = local_length(manifold, curve, T_without_boundary_points)
    lenght_cumsums = np.cumsum(local_lengths_N)  # Local lengths
    total_length = lenght_cumsums[-1] + 0.1
    # Temp is the length on the curve in the interval [0,1], i.e. move 0.1 on the length.
    temp = np.concatenate(([[0]], lenght_cumsums.reshape(-1, 1), [[total_length]]), axis=0).flatten() / total_length
    new_time = CubicSpline(temp, T)  # Here we give the proportion of length to move on the curve, and we get the time
    return curve(new_time(t))


# Returns a parametric function for the solution of the fp solver
def curve_eval_gp(Ts, T, T_min_max, Veta, dm, m, DDC, w, gp_kernel):
    # Input: Ts (Ns x?) the time knots where we want to evaluate the curve

    # If Ts is a scalar transform it into an array
    if np.isscalar(Ts):
        Ts = np.asarray([Ts])

    Ts = Ts.reshape(-1, 1)  # Ns x 1
    Ns = Ts.shape[0]
    D = DDC.shape[1]

    # The kernel parameters for the evaluation of the GP posterior mean and variance
    Ctest = np.concatenate((np.concatenate((gp_kernel.kdxddy(Ts, T), gp_kernel.kdxy(Ts, T_min_max)), axis=1),
                            np.concatenate((gp_kernel.kxddy(Ts, T), gp_kernel.kxy(Ts, T_min_max)), axis=1)),
                           axis=0)  # 2Ns + (N+2)

    dmu_mu_Ts = vec(np.concatenate((dm(Ts), m(Ts))).T) + np.kron(Ctest, Veta) @ w
    dmu_mu_Ts = dmu_mu_Ts.reshape(D, 2 * Ns, order='F')  # D x 2Ns

    dc_t = dmu_mu_Ts[:, :Ns]  # D x Ns
    c_t = dmu_mu_Ts[:, Ns:]  # D x Ns

    return c_t, dc_t


# This function vectorizes an matrix by stacking the columns
def vec(x):
    # Input: x (NxD) -> (ND x 1)
    return x.flatten('F').reshape(-1, 1)


# Master function that chooses the solver
def compute_geodesic(solver, manifold, c0, c1, solution=None):
    # solver: is an object that has all the parameters of the chosen solver ('fp', 'bvp')
    # solution: a dictionary that keeps the final parameters for each solver
    if solver.name == 'bvp':
        geodesic_solution = solver_bvp(solver, manifold, c0, c1, solution)
    elif solver.name == 'fp':
        geodesic_solution = solver_fp(solver, manifold, c0, c1, solution)
    elif solver.name == 'graph':
        geodesic_solution = solver_graph(solver, manifold, c0, c1, solution)
    elif solver.name == 'comb':
        geodesic_solution = solver_comb(solver, manifold, c0, c1, solution)
    else:
        print("TODO: Not supported solver!\n")
        sys.exit(1)

    return geodesic_solution


# This solver initializes the curve with the GRAPH if solution does not exist
def solver_comb(solver, manifold, c0, c1, solution):

    try:
        # If solution exists run SOLVER_1
        if solution is not None:
            curve, logmap, curve_length_eval, failed, solution \
                = solver_bvp(solver.solver_1, manifold, c0, c1, solution)
        # If solution does not exist run first SOLVER_2 and then SOLVER_1
        elif solution is None:
            curve, logmap, curve_length_eval, failed, solution \
                = solver_graph(solver.solver_2, manifold, c0, c1, solution)
            curve, logmap, curve_length_eval, failed, solution \
                = solver_bvp(solver.solver_1, manifold, c0, c1, solution)
    except Exception:
        print('Geodesic caught!')
        # If method failed
        curve = lambda t: evaluate_failed_solution(c0, c1, t)
        logmap = (c1 - c0)  # D x 1
        solution = None
        failed = True
        curve_length_eval = curve_length(manifold, curve)
        logmap = curve_length_eval * logmap.reshape(-1, 1) / np.linalg.norm(logmap)  # Scaling for normal coordinates

    return curve, logmap, curve_length_eval, failed, solution


# This is the default solver that is a build-in python BVP solver.
def solver_bvp(solver, manifold, c0, c1, init_solution):
    # c0, c1: Dx1
    c0 = c0.reshape(-1, 1)
    c1 = c1.reshape(-1, 1)
    D = c0.shape[0]

    # The functions that we need for the bvp solver
    ode_fun = lambda t, c_dc: second2first_order(manifold, c_dc)  # D x T, implements c'' = f(c, c')
    bc_fun = lambda ya, yb: boundary_conditions(ya, yb, c0, c1)  # 2D x 0, what returns?

    # Initialize the curve with straight line or with another given curve
    T = 30
    t_init = np.linspace(0, 1, T, dtype=np.float32)  # T x 0
    if init_solution is None:
        c_init = np.outer(c0, (1.0 - t_init.reshape(1, T))) + np.outer(c1, t_init.reshape(1, T))  # D x T
        dc_init = (c1 - c0).reshape(D, 1).repeat(T, axis=1)  # D x T
    else:
        if (init_solution['solver'] == 'fp') | (init_solution['solver'] == 'bvp') | (init_solution['solver'] == 'graph'):
            c_init, dc_init = init_solution['curve'](t_init)  # D x T, D x T
        else:
            print('The initial curve solution to the solver does not exist (bvp)!')
            sys.exit(1)
    c_dc_init = np.concatenate((c_init, dc_init), axis=0)  # 2D x T

    # Solve the geodesic problem
    result = solve_bvp(ode_fun, bc_fun, t_init.flatten(), c_dc_init, tol=solver.tol, max_nodes=solver.NMax)

    # Provide the output, if solver failed return the straight line as solution
    if result.success:
        curve = lambda t: evaluate_solution(result, t)
        logmap = result.y[D:, 0]  # D x 1
        solution = {'solver': 'bvp', 'curve': curve}
        failed = False
    else:
        print('Geodesic solver (bvp) failed!')
        curve = lambda t: evaluate_failed_solution(c0, c1, t)
        logmap = (c1 - c0)  # D x 1
        solution = None
        failed = True

    # Compute the curve length under the Riemannian measure and compute the logarithmic map
    curve_length_eval = curve_length(manifold, curve)
    logmap = curve_length_eval * logmap.reshape(-1, 1) / np.linalg.norm(logmap)  # Scaling for normal coordinates

    return curve, logmap, curve_length_eval, failed, solution


# This is the fp method solver. It can be initialized with the solution of the graph based solver.
# From the graph solver we get a set of training points c_n and we use them to infer the DDC. Note that
# the time positions (knots) of the given points c_n are not equal to the knots for the DDC.
def solver_fp(solver, manifold, c0, c1, solution=None):
    # The C_given will include the c0, c1, and all the points cn from other sources
    # The T_given will include the 0, 1, for the boundary points and all the other possible points
    # The T represents the grid for which we search the DDC, and is generally different from the T_given

    # Input: c0, c1 (Dx1)
    c0 = c0.reshape(-1, 1)
    c1 = c1.reshape(-1, 1)
    D = c0.shape[0]
    C_given = np.concatenate((c0.reshape(1, -1), c1.reshape(1, -1)), axis=0)  # 2 x D

    # The integration interval
    t_min = 0
    t_max = 1
    T_given = np.asarray([t_min, t_max]).reshape(-1, 1)  # 2 x 1

    # The parameters of the solver
    N = solver.N
    max_iter = solver.max_iter
    T = solver.T.reshape(-1, 1)  # The positions for the DDC
    tol = solver.tol
    gp_kernel = solver.gp_kernel

    # The covariance of the output dimensions
    v = c1 - c0
    Veta = ((v.T @ solver.Sdata @ v) * solver.Sdata)  # Estimate in Bayesian style the amplitude

    # Parametric functions of prior mean, dmean, ddmean
    m = lambda t: c0.T + t * (c1 - c0).T  # T x D
    dm = lambda t: np.ones((t.shape[0], 1)) * (c1 - c0).T  # T x D
    ddm = lambda t: np.zeros((t.shape[0], 1)) * (c1 - c0).T  # T x D
    dm_m_T = vec(np.concatenate((dm(T), m(T)), axis=0).T)  # 2N x D, keep a vector for speed of the m, dm on T

    # The residual/noise matrix of the GP, fixed small noise for the time knots: T, 0, 1
    I_D = np.eye(D)
    R = block_diag(solver.sigma2 * I_D)
    for n in range(N - 1):
        R = block_diag(R, solver.sigma2 * I_D)
    R = block_diag(R, 1e-10 * I_D, 1e-10 * I_D)  # (N+2)D x (N+2)D

    # If a previous solution of the curve is provided use it, otherwise initialize the parameters c''
    if solution is None:
        DDC = ddm(T)  # N x D
    elif solution['solver'] == 'fp':
        DDC = solution['ddc']
    elif solution['solver'] == 'graph':
        C_near_curve = solution['points']  # The points from the training data used in the graph
        var_extra_points = solution['noise']  # The variance (noise) of the extra points we consider for the curve

        N_near_curve = C_near_curve.shape[0]
        T_given_points = solution['time_stamps'].reshape(-1, 1)

        C_given = np.concatenate((C_given, C_near_curve), axis=0)  # Add the points on the given vector
        T_given = np.concatenate((T_given, T_given_points), axis=0)

        # Include the points near the curve coming from the other solver as information to the GP
        for n in range(N_near_curve):
            R = block_diag(R, var_extra_points * I_D)  # The noise of the new points

        # Estimate the DDC from the given data
        Ctrain_temp = gp_kernel.kddxy(T, T_given)
        Btrain_temp = gp_kernel.kxy(T_given, T_given)

        R_temp = block_diag(1e-10 * I_D, 1e-10 * I_D)  # (N+2)D x (N+2)D
        # Include the last points as information in the posterior GP
        for n in range(N_near_curve):
            R_temp = block_diag(R_temp, var_extra_points * I_D)  # The noise of the new points

        # The prior mean function and the observed values, for the given points c0, c1, c_n
        y_hat_temp = m(T_given)
        y_obs_temp = C_given

        Btrain_R_inv_temp = np.linalg.inv(np.kron(Btrain_temp, Veta) + R_temp)
        kronCtrainVeta_temp = np.kron(Ctrain_temp, Veta)

        DDC = kronCtrainVeta_temp @ (Btrain_R_inv_temp @ vec((y_obs_temp - y_hat_temp).T))
        DDC = DDC.reshape(D, N, order='F').T
    else:
        print('This solution cannot be used in this solver (fp)!')
        sys.exit(1)

    # Precompute the kernel components and keep them fixed
    Ctrain = np.concatenate((np.concatenate((gp_kernel.kdxddy(T, T), gp_kernel.kdxy(T, T_given)), axis=1),
                             np.concatenate((gp_kernel.kxddy(T, T), gp_kernel.kxy(T, T_given)), axis=1)),
                            axis=0)  # 2N x (N+2)

    Btrain = np.concatenate((np.concatenate((gp_kernel.kddxddy(T, T), gp_kernel.kddxy(T, T_given)), axis=1),
                             np.concatenate((gp_kernel.kxddy(T_given, T), gp_kernel.kxy(T_given, T_given)), axis=1)),
                            axis=0)  # (N+2) x (N+2)

    # The evaluation of the prior m, ddm on T, and the observed values
    y_hat = np.concatenate((ddm(T), m(T_given)), axis=0)  # N+2 x D
    y_obs = lambda ddc: np.concatenate((ddc, C_given), axis=0)  # N+2 x D

    # Define the posterior mean for the knots t_n, parametrized by the ddc_n
    Btrain_R_inv = np.linalg.inv(np.kron(Btrain, Veta) + R)  # Precompute for speed
    kronCtrainVeta = np.kron(Ctrain, Veta)
    dmu_mu_post = lambda ddc: dm_m_T + kronCtrainVeta @ (Btrain_R_inv @ vec((y_obs(ddc) - y_hat).T))

    # Solve the geodesic problem as a fixed-point iteration
    iteration = 1
    convergence_cond = 0
    while True:
        # The current posterior mean and dmean on the knots t_n for the parameters DDC
        dmu_mu_post_curr = dmu_mu_post(DDC)

        # Separate the m and dm and then reshape
        dmu_mu_post_curr_temp = dmu_mu_post_curr.reshape(D, 2 * N, order='F')  # D x N
        dmu_post_curr = dmu_mu_post_curr_temp[:, :N]  # D x N
        mu_post_curr = dmu_mu_post_curr_temp[:, N:]  # D x N

        DDC_curr = geodesic_system(manifold, mu_post_curr, dmu_post_curr).T  # N x D
        cost_curr = (DDC - DDC_curr) ** 2
        condition_1 = (cost_curr < tol).all()  # Check point-wise if lower than tol
        condition_2 = (iteration > max_iter)
        if condition_1 | condition_2:
            if condition_1:
                convergence_cond = 1
            if condition_2:
                convergence_cond = 2
            break

        # The gradient for the update
        grad = DDC - DDC_curr

        # Search for optimal step-size
        alpha = 1.0
        for i in range(3):
            DDC_temp = DDC - alpha * grad

            dmu_mu_post_curr_temp = dmu_mu_post(DDC_temp).reshape(D, 2 * N, order='F')
            dmu_post_curr_temp = dmu_mu_post_curr_temp[:, :N]  # D x N
            mu_post_curr_temp = dmu_mu_post_curr_temp[:, N:]  # D x N

            cost_temp = (DDC_temp - geodesic_system(manifold, mu_post_curr_temp, dmu_post_curr_temp).T) ** 2  # N x D
            if cost_temp.sum() < cost_curr.sum():
                break
            else:
                alpha = alpha * 0.33

        # Update the parameters
        DDC = DDC - alpha * grad
        iteration = iteration + 1

    # Prepare the output
    if convergence_cond == 2:
        print('Geodesic solver (fp) failed!')
        curve = lambda t: evaluate_failed_solution(c0, c1, t)
        logmap = (c1 - c0).flatten()  # (D,)
        failed = True
        solution = None
    elif convergence_cond == 1:
        w = (Btrain_R_inv @ vec((y_obs(DDC) - y_hat).T))  # The posterior weights
        curve = lambda t: curve_eval_gp(t, T, T_given, Veta, dm, m, DDC, w, gp_kernel)
        logmap = curve(0)[1].flatten()  # (D,)
        failed = False
        # Use a return variable for debugging
        solution = {'ddc': DDC, 'total_iter': iter, 'cost': cost_curr, 'curve': curve, 'solver': 'fp'}
    elif convergence_cond == 0:
        failed = True
        print('Geodesic solve (FP) failed for some reason!')

    # Compute the curve length and scale the unit logmap
    curve_length_eval = curve_length(manifold, curve)
    logmap = curve_length_eval * logmap.reshape(-1, 1) / np.linalg.norm(logmap)  # Scaling for normal coordinates.

    return curve, logmap, curve_length_eval, failed, solution


# An approximate graph based solver and a cubic spline result
def solver_graph(solver, manifold, c0, c1, solution=None):

    # The weight matrix
    W = solver.New_Graph.todense()

    # Find the Euclidean closest points on the graph to be used as fake start and end.
    _, c0_indices = solver.kNN_graph.kneighbors(c0.T)  # Find the closest kNN_num+1 points to c0
    _, c1_indices = solver.kNN_graph.kneighbors(c1.T)  # Find the closest kNN_num+1 points to c1
    ind_closest_to_c0 = np.nan  # The index in the training data closer to c0
    ind_closest_to_c1 = np.nan
    cost_to_c0 = 1e10
    cost_to_c1 = 1e10
    for n in range(solver.kNN_num - 1):  # We added one extra neighbor when we constructed the graph

        # Pick the next point in the training data tha belong in the kNNs of c0 and c1
        ind_c0 = c0_indices[0, n]  # kNN index from the training data
        ind_c1 = c1_indices[0, n]  # kNN index from the training data

        x_c0 = solver.data[ind_c0, :].reshape(-1, 1)  # The kNN point near to c0
        x_c1 = solver.data[ind_c1, :].reshape(-1, 1)  # The kNN point near to c1

        # Construct temporary straight lines
        temp_curve_c0 = lambda t: evaluate_failed_solution(c0, x_c0, t)
        temp_curve_c1 = lambda t: evaluate_failed_solution(c1, x_c1, t)

        # Shortest path on graph prefers "low-weight" connections
        temp_cost_c0 = curve_length(manifold, temp_curve_c0, tol=solver.tol)
        temp_cost_c1 = curve_length(manifold, temp_curve_c1, tol=solver.tol)

        # We found one of the  Euclidean kNNs that has closer Riemannian distance from the other kNNs we have checked.
        if temp_cost_c0 < cost_to_c0:
            ind_closest_to_c0 = ind_c0
            cost_to_c0 = temp_cost_c0

        if temp_cost_c1 < cost_to_c1:
            ind_closest_to_c1 = ind_c1
            cost_to_c1 = temp_cost_c1

    # The closest points in the graph to the test points c0, c1
    source_ind = ind_closest_to_c0
    end_ind = ind_closest_to_c1

    path = [end_ind]
    pairwise_lengths = []
    temp_ind = end_ind

    # Find the discrete path between source and sink. Each cell [i,j] keeps the previous point path before reaching j from i
    while True:
        prev_ind = solver.predecessors[source_ind, temp_ind]  # The previous point to reach the [goal == temp_ind]
        if prev_ind == -9999:  # There is not any other point in the path
            break
        else:
            path.append(prev_ind)
            pairwise_lengths.append(W[temp_ind, prev_ind])  # Weight/distance between the current and previous node
            temp_ind = prev_ind  # Move the pointer to one point close to the source.

    path.reverse()  # Reverse the path from [end, ..., source] -> [source, ..., end]
    inds = np.asarray(path)

    DiscreteCurve_data = solver.data[inds.flatten(), :]  # The discrete path on the graph

    # A heuristic to smooth the discrete path with a mean kernel
    DiscreteCurve_data = np.concatenate((c0.T, DiscreteCurve_data[1:-1, :], c1.T), axis=0)
    DiscreteCurve_new = np.empty((0, c0.shape[0]))
    for n in range(1, DiscreteCurve_data.shape[0]-1):
        new_point = (DiscreteCurve_data[n-1] + DiscreteCurve_data[n+1] + DiscreteCurve_data[n]) / 3
        DiscreteCurve_new = np.concatenate((DiscreteCurve_new, new_point.reshape(1, -1)), axis=0)
    DiscreteCurve_data = DiscreteCurve_new.copy()
    DiscreteCurve = np.concatenate((c0.T, DiscreteCurve_data, c1.T), axis=0)

    # Simple time parametrization of the curve
    N_points = DiscreteCurve.shape[0]  # Number of points in the discrete shortest path
    t = np.linspace(0, 1, num=N_points, endpoint=True)  # The time steps to construct the spline

    # Interpolate the points with a cubic spline.
    curve_spline = CubicSpline(t, DiscreteCurve)  # The continuous curve that interpolates the points on the graph
    dcurve_spline = curve_spline.derivative()  # The derivative of the curve
    curve = lambda t: evaluate_spline_solution(curve_spline, dcurve_spline, t)

    # Return the solution
    solution = {'curve': curve, 'solver': solver.name,
                'points': DiscreteCurve[1:-1, :], 'time_stamps': t[1:-1]}
    curve_length_eval = curve_length(manifold, curve, tol=solver.tol, limit=solver.limit)
    logmap = dcurve_spline(0).reshape(-1, 1)  # The initial velocity
    logmap = curve_length_eval * logmap.reshape(-1, 1) / np.linalg.norm(logmap)  # Scaling for normal coordinates.
    failed = False

    return curve, logmap, curve_length_eval, failed, solution


# This function implements the exponential map
def expmap(manifold, x, v):
    # Input: v,x (Dx1)
    x = x.reshape(-1, 1)
    v = v.reshape(-1, 1)
    D = x.shape[0]

    ode_fun = lambda t, c_dc: second2first_order(manifold, c_dc).flatten()  # The solver needs this shape (D,)
    if np.linalg.norm(v) > 1e-5:
        curve, failed = solve_expmap(manifold, x, v, ode_fun)
    else:
        curve = lambda t: (x.reshape(D, 1).repeat(np.size(t), axis=1),
                           v.reshape(D, 1).repeat(np.size(t), axis=1))  # Return tuple (2D x T)
        failed = True

    return curve, failed


# This function solves the initial value problem for the implementation of the expmap
def solve_expmap(manifold, x, v, ode_fun):
    # D = x.shape[0]
    # The vector now is in normal coordinates
    required_length = np.linalg.norm(v)  # The tangent vector lies in the normal coordinates

    # Rescale the vector to be proper for solving the geodesic.
    v = v / required_length
    if manifold.is_diagonal():
        M = np.diag(manifold.metric_tensor(x).flatten())
    elif not manifold.is_diagonal():
        M = manifold.metric_tensor(x)[0]

    a = (required_length / np.sqrt(v.T @ M @ v))

    # The vector now is on the exponential coordinates
    v = a * v

    init = np.concatenate((x, v), axis=0).flatten()  # 2D x 1 -> (2D, ), the solver needs this shape
    failed = False

    # Solve the IVP problem
    solution = solve_ivp(ode_fun, [0, 1], init, dense_output=True)  # First solution of the IVP problem
    curve = lambda t: evaluate_solution(solution, t)  # with length(c(t)) != ||v||_c
    solution_length = curve_length(manifold, curve, 0, 1)  # the length the geodesic should have

    # Note: This is new
    if (solution_length - required_length)**2 > 1e-2:
        failed = True

    return curve, failed


class SolverComb:

    def __init__(self, solver_1, solver_2):
        self.name = 'comb'
        self.solver_1 = solver_1
        self.solver_2 = solver_2


class SolverGraph:

    def __init__(self, manifold, data, kNN_num, tol=1e-5, limit=50):
        self.manifold = manifold
        self.data = data
        self.kNN_num = kNN_num + 1  # The first point for the training data is always the training data point.
        self.kNN_graph = NearestNeighbors(n_neighbors=kNN_num + 1, algorithm='ball_tree').fit(data)  # Find the nearest neighbors
        self.tol = tol
        self.limit = limit
        N_data = data.shape[0]

        # Find the Euclidean kNN
        distances, indices = self.kNN_graph.kneighbors(data)
        Weight_matrix = np.zeros((N_data, N_data))  # The indicies of the kNN for each data point
        for ni in range(N_data):  # For all the data
            p_i = data[ni, :].reshape(-1, 1)  # Get the data point
            kNN_inds = indices[ni, 1:]  # Find the Euclidean kNNs

            for nj in range(kNN_num):  # For each Euclidean kNN connect with the Riemannian distance
                ind_j = kNN_inds[nj]  # kNN index
                p_j = data[ind_j, :].reshape(-1, 1)  # The kNN point
                temp_curve = lambda t: evaluate_failed_solution(p_i, p_j, t)
                # Note: Shortest path on graph prefers "low-weight" connections
                Weight_matrix[ni, ind_j] = curve_length(manifold, temp_curve, tol=tol, limit=limit)

            if ni % 100 == 0:
                print("[Initialize Graph] [Processed point: {}/{}]".format(ni, N_data))

        # Make the weight matrix symmetric
        Weight_matrix = 0.5 * (Weight_matrix + Weight_matrix.T)

        self.New_Graph = csr_matrix(Weight_matrix, shape=(N_data, N_data))

        # Find the shortest path between all the points
        self.dist_matrix, self.predecessors = \
            shortest_path(csgraph=self.New_Graph, directed=False, return_predecessors=True)

        self.name = 'graph'


# This class is used to define an object for the bvp solver, and this object holds the parameters of the solver.
class SolverBVP:

    def __init__(self, NMax=1000, tol=1e-1):
        self.NMax = NMax
        self.tol = tol
        self.name = 'bvp'


# This class is used to define an object for the fp solver, and this object holds the parameters of the solver.
class SolverFP:

    def __init__(self, D=None, N=10, tol=1e-1, max_iter=1000, sigma=1e-4, ell=None, Sdata=None, kernel_type=None):

        if D is None:
            print("Dimensionality of the space has to be given for the solver!\n")
            sys.exit(1)
        else:
            self.D = D

        self.N = N
        self.tol = tol
        self.T = np.linspace(0, 1, N).reshape(-1, 1)  # N x 1
        self.max_iter = max_iter
        self.sigma2 = sigma ** 2

        if ell is None:
            self.ell = np.sqrt(0.5 * (self.T[1] - self.T[0]))
        else:
            self.ell = ell

        if Sdata is None:
            self.Sdata = np.eye(self.D)
        else:
            self.Sdata = Sdata

        if kernel_type is None:
            print("Kernel type has not been specified (default: Squared Exponential).\n")

        self.gp_kernel = SE_Kernel(self.ell, alpha=1.0)
        self.name = 'fp'


# This class is used to define an object for the GP kernel, and holds the parameters and functions of the GP kernel.
class SE_Kernel:

    def __init__(self, ell, alpha=1.0):
        self.ell = ell
        self.ell2 = ell ** 2
        self.alpha = alpha

    def kxy(self, x, y):
        # x,y: N x 1
        dist2 = (x.reshape(-1, 1) - y.reshape(1, -1)) ** 2
        K = self.alpha * np.exp(- (0.5 / self.ell2) * dist2)
        return K

    def kdxy(self, x, y):
        Kdxy = -self.delta(x, y) * self.kxy(x, y)
        return Kdxy

    def kxdy(self, x, y):
        Kxdy = - self.kdxy(x, y)
        return Kxdy

    def kdxdy(self, x, y):
        Kdxdy = (1.0 / self.ell2 - self.delta(x, y) ** 2) * self.kxy(x, y)
        return Kdxdy

    def kxddy(self, x, y):
        Kxddy = - self.kdxdy(x, y)
        return Kxddy

    def kdxddy(self, x, y):
        Kdxddy = - self.kddxdy(x, y)
        return Kdxddy

    def kddxy(self, x, y):
        Kddxy = - self.kdxdy(x, y)
        return Kddxy

    def kddxdy(self, x, y):
        Kddxdy = (-3.0 * self.delta(x, y) / self.ell2 + self.delta(x, y) ** 3) * self.kxy(x, y)
        return Kddxdy

    def kddxddy(self, x, y):
        Kddxddy = (self.delta(x, y) ** 4 - 6 * (self.delta(x, y) ** 2) / self.ell2 + 3 / (self.ell2 ** 2)) * self.kxy(x,
                                                                                                                      y)
        return Kddxddy

    def delta(self, x, y):
        d = (x.reshape(-1, 1) - y.reshape(1, -1)) / self.ell2
        return d
