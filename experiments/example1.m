% ================================================ %
% === Non-parametric Riemannian metric example === %
% ================================================ %
%
% A non-parametric Riemannian metric is defined in the space where the data
% lie. Then we are able to compute shortest paths between points by solving 
% the system of nonlinear ordinary differential equations.
%
% See for details:
% "A Locally Adaptive Normal Distribution",
%   G. Arvanitidis, L. K. Hansen, S. Hauberg,
%   Neural Information Processing Systems (NeurIPS) 2016.
%
% "Latent Space Oddity: on the Curvature of Deep Generative Models",
%   G. Arvanitidis, L. K. Hansen, S. Hauberg,
%   International Conference on Learning Representations (ICLR) 2018.
%
% Author: Georgios Arvanitidis

clear;
close all;

% Generate some random data
load /data/example1_data_nonparametric.mat;
N = size(data, 1);

% Here we construct a nonparametric Riemannian metric
sigma_manifold = 0.1;
rho_manifold = 0.01;
manifold = diagonal_local_pca_metric(data, sigma_manifold, rho_manifold);

% Pick randomly two points
c0 = data(200,:)'; c1 = data(1, :)';

% Define the Fixed-Point shortest path solver with default parameters
fp_options.Sdata = cov(data);
fp_options.N = 10;
D = 2;
% fp_options.tol = 1e-2;
solver_fp = geodesic_solver_fp(D, fp_options);

% Define the bvp5 solver with default options
bvp5c_options = [];
% bvp5c_options.tol = 1e-3;
solver_bvp5c = geodesic_solver_bvp5c(bvp5c_options);

% Compute the shortest paths (?he bvp5c is initialized with the fp solution).
[curve_fp, logmap_fp, len_fp, failed_fp, solution_fp] = compute_geodesic( solver_fp, manifold, c0, c1);
[curve_bvp5c, logmap_bvp5c, len_bvp5c, failed_bvp5c, solution_bvp5c] = compute_geodesic( solver_bvp5c, manifold, c0, c1, curve_fp);

% Compute the exponential maps
[ curve_expmap_fp, solution_expmap_fp] = expmap(manifold, c0, logmap_fp);
[ curve_expmap_bvp5c, solution_expmap_bvp5c] = expmap(manifold, c0, logmap_bvp5c);

% Plot the shortest paths
figure;
scatter(data(1:end,1),data(1:end,2),30,'filled'); hold on; grid on; axis equal;
h_fp = plot_curve(curve_fp, 'g', 'LineWidth',2); 
h_bvp5c = plot_curve(curve_bvp5c, 'r', 'LineWidth',2); 
legend([h_fp, h_bvp5c], 'Fixed-Point', 'bvp5c');
hq_bvp5c = quiver(c0(1),c0(2),logmap_bvp5c(1),logmap_bvp5c(2),0.5); hq_bvp5c.Color = 'r';
hq_fp = quiver(c0(1),c0(2),logmap_fp(1),logmap_fp(2),0.5); hq_fp.Color = 'g';
title('Compare shortest paths and logarithmic maps');

% Plot the shortest path vs exponential maps (fixed-point method)
figure;
scatter(data(1:end,1),data(1:end,2),30,'filled'); hold on; grid on; axis equal;
h_fp_path = plot_curve(curve_fp, 'g', 'LineWidth',2); 
h_fp_expmap = plot_curve(curve_expmap_fp, '--k', 'LineWidth',2); 
legend([h_fp_path, h_fp_expmap], 'Shortest Path', 'Exponential Map');
title('Compare shortest path with exponential map (fixed-point)');

% Plot the shortest path vs exponential maps (bvp5c method)
figure;
scatter(data(1:end,1),data(1:end,2),30,'filled'); hold on; grid on; axis equal;
h_bvp5c_path = plot_curve(curve_bvp5c, 'r', 'LineWidth',2); 
h_bvp5c_expmap = plot_curve(curve_expmap_bvp5c, '--k', 'LineWidth',2); 
legend([h_bvp5c_path, h_bvp5c_expmap], 'Shortest Path', 'Exponential Map');
title('Compare shortest path with exponential map (bvp5c)');

% Now compute all the geodesics
pause(0.01);
mu = [0; 1]; % set the mean point
[Curves_bvp5c, Logs_bvp5c, Lens_bvp5c, Fails_bvp5c, Solutions_bvp5c, time_elapsed_bvp5c] = compute_all_geodesics (solver_bvp5c, manifold, mu, data);
[Curves_fp, Logs_fp, Lens_fp, Fails_fp, Solutions_fp, time_elapsed_fp] = compute_all_geodesics (solver_fp, manifold, mu, data);

% Plot the results
figure;
scatter(data(1:end,1),data(1:end,2),30,'filled'); hold on; grid on; axis equal;
for n = 1:N
    if(~Fails_bvp5c(n))
        plot_curve(Curves_bvp5c{n}, 'r', 'LineWidth', 0.5);
    else
        plot_curve(Curves_bvp5c{n}, '--k', 'LineWidth', 0.5);
    end
end
title('All shortest paths for (bvp5c)');

figure;
scatter(data(1:end,1),data(1:end,2),30,'filled'); hold on; grid on; axis equal;
for n = 1:N
    if(~Fails_fp(n))
        plot_curve(Curves_fp{n}, 'g', 'LineWidth', 0.5);
    else
        plot_curve(Curves_fp{n}, '--k', 'LineWidth', 0.5);
    end
end
title('All shortest paths for (fp)');

% Compare the running time and number of failures.
fprintf('===============\n')
fprintf('=== Results ===\n')
fprintf('===============\n')
fprintf('Running times:\n - fixed point: %f\n - bvp5c: %f\n',[time_elapsed_fp; time_elapsed_bvp5c])
fprintf('Number of fails:\n - fixed point: %d\n - bvp5c: %d\n',[sum(Fails_fp); sum(Fails_bvp5c)])
