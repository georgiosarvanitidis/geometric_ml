function ddc = geodesic_system(manifold, c, dc)
% This function defines the ODE to be solved while searching for shortest
% paths on Riemannian manifolds. The ODE in question is defined as in the
% "Latent Space Oddity: on the Curvature of Deep Generative Models", 
%   International Conference on Learning Representations (ICLR) 2018.
%
% Inputs: 
%   manifold - any class which implements the metric_tensor method.
%   c        - DxN matrix of N locations c(t_1), ..., c(t_N) of
%              D-dimensional points
%   dc       - DxN matrix of N locations c'(t_1), ..., c'(t_N) of
%              D-dimensional derivatives at the specified locations
%
% Output: 
%   ddc      - DxN matrix of second derivatives of c at locations t_n.
%
% Author: Georgios Arvanitidis, Soren Hauberg
  

  
    [D, N] = size(c);
    if (size(dc, 1) ~= D || size(dc, 2) ~= N)
        error('geodesic_system: second and third input arguments must have same dimensionality');
    end % if
  
    %% Evaluate metric
    [M, dM] = metric_tensor(manifold, c);
    
    %% Prepare the output
    ddc = zeros(D, N); % D x N
        
    %% Separate cases diagonal or not metric
    if (isdiagonal(manifold))
        % Each row is dM = [dM/dc1, dM/dc2, ..., dM/dcD], where each column
        % dM/dcj is the derivatives of the diagonal metric elements wrt cj.
        % This has dimension Dx1.
        
        for n = 1:N
            dMn = reshape(dM(n, :, :), D, D); % D x D

            ddc(:,n) = -0.5*(2*(dMn.*dc(:,n))*dc(:,n) - dMn'*(dc(:,n).^2))./(M(n, :)');
        end % for
    else
        % Each row is dM = [dM/dc1| dM/dc2| ...| dM/dcD], where each slice
        % dM/dcj is the derivatives of the metric elements wrt cj.
        % This has dimension (DxD)xD.
        
        for n = 1:N
        
            M_n = reshape(M(n, :, :), D, D); 
            if(rcond(M_n) < 1e-15)
                disp('Bad Condition Number of the metric');
                error('Bad Condition Number of the metric');
            end

            % This is the dvec[M]/dc, 
            % Each slice dM(n, :, :, d) is the derivative dM/dc_d for the nth point
            dvecMdc_n = reshape(dM(n, :, :, :), D*D, D); % (D*D)xD

            % Construct the block diagonal matrix
            blck = kron(eye(D), dc(:, n)');

            % This is the differential equation
            ddc(:, n) = -0.5 * ( M_n \ (2 * blck * dvecMdc_n * dc(:,n) - dvecMdc_n' * kron(dc(:,n), dc(:,n))));
        end % for
    end % if
end % function