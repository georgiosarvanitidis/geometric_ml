function [ gp_kernel ] = se_kernel( ell, alpha )
% Construct an object of a squared exponential kernel defined as
%   k(x, y) = alpha * exp( -0.5 (x - y)^2 / ell^2 )
    
    % Prepare the struct.
    gp_kernel = struct();
    
    % Keep the lengthscale parameter.
    gp_kernel.ell = ell;
    gp_kernel.ell2 = ell^2;
    
    % Keep the kernel amplitude parameter.
    if( nargin > 1)
        gp_kernel.alpha = alpha;
    else
        gp_kernel.alpha = 1.0;
    end
    
    gp_kernel = class(gp_kernel, 'se_kernel'); 
end

