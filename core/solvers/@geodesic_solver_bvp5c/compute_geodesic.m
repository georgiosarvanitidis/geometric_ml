function [ curve, logmap, len, failed, solution ] = compute_geodesic( solver, manifold, p0, p1, init_curve )
% This function computes the shortest path between two given points on a
% Riemannian manifold. The utilized method is MATLAB's 'bvp5c'.
%
% Input:
%   solver     - The object from the sovler class.
%   manifold   - An object of any manifold class.
%   p0, p1     - The stparting and ending point of the curve Dx1.
%   init_curve - Starting solution for the parametric curve, c(t):[0,1]->M.
%
% Output:
%   curve    - A parametric curve c(t):[0,1]->M.
%   logmap   - The logarithmic map Dx1.
%   len      - A scalar for the curve length.
%   failed   - A boolean, if solver failed this value is true.
%   solution - The struct containing the details of the solution.
%   
% Author: Soren Hauberg, Georgios Arvanitidis

    %% Format input
    p0 = p0(:); % Dx1
    p1 = p1(:); % Dx1
    D = numel(p0);

    %% Set up boundary conditions
    bc = @(a, b) boundary(a, b, p0, p1);
    
    %% Set up ODE system
    odefun = @(x, y) second2first_order(manifold, y);

    %% Set up the options
    options = bvpset('Vectorized', 'on', ...
                     'RelTol', solver.tol, ...
                     'NMax', solver.NMax);
    
    %% Initial guess
    if (nargin < 5)
        x_init = linspace(0, 1, 10); % 1xT
        y_init = @(t) init_solution(p0, p1, t);
        init = bvpinit(x_init, y_init);
    else
        x_init = linspace(0, 1, 10); % 1xT
        y_init = @(t) init_solution_given(init_curve, t);
        init = bvpinit(x_init, y_init);
    end % if
    
    %% Try to solve the ODE
    tic;
    try
%         solution = bvp4c(odefun, bc, init, options);
        solution = bvp5c(odefun, bc, init, options);
        if (isfield(solution, 'stats') && isfield(solution.stats, 'maxerr'))
          if (isfield(options, 'RelTol') && isscalar(options.RelTol))
            reltol = options.RelTol;
          else
            reltol = 1e-3;
          end % if
          failed = (solution.stats.maxerr > reltol);
        else
          failed = false;
        end % if
    catch
      disp('Geodesic solver (bvp5c) failed!');
      failed = true;
      solution = [];
    end % try
    solution.time_elapsed = toc;
    
    %% Provide the output
    if (failed)
        curve = @(t) evaluate_failed_solution(p0, p1, t);
        logmap = (p1 - p0);
    else
        curve = @(t) evaluate_solution(solution, t, 1);
        logmap = solution.y((D+1):end, 1);
    end % if
    if (nargout > 1)
        len = curve_length(manifold, curve);
        logmap = len * logmap / norm(logmap);
    end % if
    
end % function
 
%% Additional functions 
function bc = boundary(p0, p1, p0_goal, p1_goal)
  D = numel(p0_goal);
  d1 = p0(1:D) - p0_goal(:);
  d2 = p1(1:D) - p1_goal(:);
  bc = [d1; d2];
end % function

function [c, dc] = evaluate_failed_solution(p0, p1, t)
  t = t(:); % 1xT
  c = (1 - t) * p0.' + t * p1.'; % TxD
  dc = repmat ((p1 - p0).', numel(t), 1); % TxD
  c = c.'; dc = dc.';
end % function

function state = init_solution(p0, p1, t)
  [c, dc] = evaluate_failed_solution(p0, p1, t);
  state = cat(1, c, dc); % 2DxT
end % function

function state = init_solution_given(solution, t)
    [c, dc] = solution(t);
    state = cat(1, c, dc); % 2DxT
end % function
