function [curve, solution, failed] = expmap(manifold, mu, v)
% Compute the exponential map of v expressed in the tangent space at mu.
% The used ODE is in
%   "Latent Space Oddity: on the Curvature of Deep Generative Models", 
%       International Conference on Learning Representations (ICLR) 2018.
%
% Inputs:
%   manifold - any class which implements the metric_tensor method.
%     mu     - the origin of the tangent space. This should be a D-vector.
%     v      - a D-vector expressed in the tangent space at mu. This vector
%              is often a linear combination of points produced by the
%              'compute_geodesic' function.
%
%   Output:
%     curve    - a parametric function c:[0,1] -> M
%     solution - the output structure from the ODE solver. This is mostly 
%                used for debugging.
%
% Author: Soren Hauberg, Georgios Arvanitidis
    
  mu = mu (:);
  v = v (:);

  %% Set up ODE system
  odefun = @(x, y) second2first_order(manifold, y);
  
  %% Solve system
  if (norm(v) > 1e-5)
    [curve, solution, failed] = solve_ode (odefun, mu, v, manifold);
  else
    curve = @(t) repmat(mu(:), 1, numel(t));
    solution = struct();
    failed = false;
  end % if
end % function

%% The actual intial value problem solver
function [curve, solution, failed] = solve_ode (odefun, mu, v, manifold)
  %% Compute how long the geodesic should be
  D = numel (mu);
  req_len = norm (v);
  init = [mu(:); v(:)];
  
  %% Compute initial geodesic
  prev_t = 0;
  t = 1;
  solution = ode45 (odefun, [0, t], init);
  curve = @(tt) evaluate_solution (solution, tt, 1);
  sol_length = curve_length (manifold, curve, 0, t);
  
  %% Keep extending the geodesic until it is longer than required
  max_iter = 1000;
  for k = 1:max_iter
    if (sol_length > req_len)
      break;
    end % if
    prev_t = t;
    t = 1.1 * (req_len / sol_length) * t;
    solution = odextend (solution, odefun, t);
    curve = @(tt) evaluate_solution (solution, tt, 1);
    sol_length = curve_length (manifold, curve, 0, t);
  end % for
  
  if (sol_length > req_len)
    %% Shorten the geodesic to have the required length
    t = fminbnd (@(tt) (curve_length (manifold, curve, 0, tt) - req_len).^2, prev_t, t); % this seems to be faster than the root seeking below. Odd...
    %t = fzero (@(tt) curve_length (manifold, curve, 0, tt) - req_len, [prev_t, t]);
    
    %% Create the final solution
    curve = @(tt) evaluate_solution (solution, tt, t);
    failed = false;
  else
    failed = true;
    warning ('expmap: unable to make the solution long enough');
  end % if
end % function


