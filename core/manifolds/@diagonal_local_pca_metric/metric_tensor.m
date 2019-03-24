function [M, dMdc] = metric_tensor(manifold, c)
% This function evaluates the Riemannian metric as well as its derivative.
%
% See for details:
% "A Locally Adaptive Normal Distribution",
%   G. Arvanitidis, L. K. Hansen, S. Hauberg,
%   Neural Information Processing Systems (NeurIPS) 2016.
%
% Input:
%   c - point D x N
%
% Output:
%   M    - A matrix NxD, each row has the diagonal elements of the metric.
%   dMdc - A matrix NxDxD, the M(n,:,j) are the dM(z)/dz_j.
%
% Author: Soren Hauberg, Georgios Arvanitidis

  %% Get problem dimensions
  X = manifold.X; % RxD
  [R, D] = size(X);
  N = size(c, 2);
  sigma2 = manifold.sigma2; % scalar
  rho = manifold.rho; % scalar

  %% Handle each point
  M = NaN(N, D); % NxD
  if (nargout > 1)
    dMdc = zeros(N, D, D);
  end % if
  
  for n = 1:N
    %% Compute metric
    cn = c(:, n); % Dx1
    delta = bsxfun(@minus, X, cn.'); % RxD
    delta2 = delta.^2; % RxD
    dist2 = sum(delta2, 2); % Rx1
    w_n = exp(-0.5 * dist2 / sigma2) / ((2*pi*sigma2)^(D/2)); % Rx1
    S = (delta.^2).' * w_n + rho; % Dx1
    m = 1 ./ S; % Dx1
    M(n ,:) = m;
    
    %% Compute derivative of metric?
    if (nargout > 1)
      dSdc = 2 * diag(delta.' * w_n); % DxD
      weighted_delta = bsxfun(@times, (w_n ./ sigma2), delta); % RxD
      dSdc = dSdc - weighted_delta.' * delta2; % DxD
      dMdc(n, :, :) = bsxfun(@times, dSdc', m.^2);
    end % if
  end % for
end % function

