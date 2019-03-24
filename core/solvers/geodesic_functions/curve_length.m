function len = curve_length(manifold, curve, a, b, tol)
% This function computes the length of curve along the Riemannian manifold.
% The input curve must be parametric type: curve = @(t)...
%
% Input:
%   manifold - An object of any manifold class.
%   curve    - A parametric curve c(t):[0,1]->M.
%   a, b     - The interval boundaries.
%   tol      - If the interal is smaller than tol then do not integrate.
%
% Output:
%   len      - A scalar, the curve length on the Riemannian manifold.
%
% Author: Soren Hauberg, Georgios Arvanitidis

  %% Supply default arguments
  if (nargin < 3)
    a = 0;
  end % if
  if (nargin < 4)
    b = 1;
  end % if
  if (nargin < 5)
    tol = 1e-6;
  end % if
  
  %% Integrate
  if (abs(a - b) > 1e-6)
    if (~isnumeric(curve))
      len = integral(@(t) local_length(manifold, curve, t), a, b, 'AbsTol', tol);
    end % if
  end % if

end % function

%% The infinitesimal local lenght at a point c(t).
function d = local_length(manifold, curve, t)
  [c, dc] = curve(t); % [DxT, DxT]
  M = metric_tensor(manifold, c); % TxD or TxDxD
  if (isdiagonal(manifold))
    d = sqrt(sum(M.' .* (dc.^2), 1));
  else
    v1 = squeeze(sum(bsxfun(@times, M, dc.'), 2)); % TxD
    d = sqrt(sum(dc .* v1.', 1)); % Tx1
  end % if
end % function
