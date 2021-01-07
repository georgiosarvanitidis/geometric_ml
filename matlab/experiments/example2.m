% ============================================ %
% === Parametric Riemannian metric example === %
% ============================================ %
%
% A stochastic generator defines a stochastic Riemannian metric in the
% latent space. Then using the expected metric we are able to compute
% shortest paths between points in the latent space solving the system of
% nonlinear ordinary differential equations.
%
% See for details:
% "Latent Space Oddity: on the Curvature of Deep Generative Models",
%   G. Arvanitidis, L. K. Hansen, S. Hauberg,
%   International Conference on Learning Representations (ICLR) 2018.
%
% Author: Georgios Arvanitidis

clear;
close all;

% load the data
load ../data/example2_data_parametric.mat

% The parameters of the expected metric
muNet.W0 = W0';
muNet.W1 = W1';
muNet.W2 = W2';
muNet.b0 = b0';
muNet.b1 = b1';
muNet.b2 = b2';

sigmaNet.Wrbf = Wrbf';
sigmaNet.gammas = gammas';
sigmaNet.centers = centers;
sigmaNet.zeta = zeta;

% The manifold structure
manifold = deep_mlp(muNet, sigmaNet);

% Data plot
scatter(data(1:5:end,1), data(1:5:end,2), 30, 'filled'); hold on; grid on; axis equal;

% Pick randomly two points
c0 = data(randsample(size(data,1),1),:)'; c1 = data(randsample(size(data,1),1), :)';

% Define the Fixed-Point shortest path solver with default parameters
fp_options.Sdata = cov(data);
fp_options.N = 10;
dim = 2;
% fp_options.tol = 1e-2;
solver_fp = geodesic_solver_fp(dim, fp_options);

% Define the bvp5 solver with default options
bvp5c_options = [];
% bvp5c_options.tol = 1e-3;
solver_bvp5c = geodesic_solver_bvp5c(bvp5c_options);

% Compute the shortest paths
[curve_fp, logmap_fp, len_fp, failed_fp, solution_fp] = compute_geodesic( solver_fp, manifold, c0, c1);
[curve_bvp5c, logmap_bvp5c, len_bvp5c, failed_bvp5c, solution_bvp5c] = compute_geodesic( solver_bvp5c, manifold, c0, c1);

% Compute the exponential maps
[ curve_expmap_fp, solution_expmap_fp] = expmap(manifold, c0, logmap_fp);
[ curve_expmap_bvp5c, solution_expmap_bvp5c] = expmap(manifold, c0, logmap_bvp5c);

% Plot the results
h_fp = plot_curve(curve_fp, 'g', 'LineWidth',2); 
h_bvp5c = plot_curve(curve_bvp5c, 'r', 'LineWidth',2); 
legend([h_fp, h_bvp5c], 'Fixed-Point', 'bvp5c');
hq_bvp5c = quiver(c0(1),c0(2),logmap_bvp5c(1),logmap_bvp5c(2),0.1); hq_bvp5c.Color = 'r';
hq_fp = quiver(c0(1),c0(2),logmap_fp(1),logmap_fp(2),0.1); hq_fp.Color = 'g';
title('Compare shortest paths and logarithmic maps');

% Compare the running time and curve lenths
fprintf('===============\n')
fprintf('=== Results ===\n')
fprintf('===============\n')
fprintf('Running times:\n - fixed point: %f\n - bvp5c: %f\n',[solution_fp.time_elapsed; solution_bvp5c.time_elapsed])
