function [ K ] = kxy( gp_kernel, x, y )

    dist2 = (x(:) - y(:)').^2; % N x N
    K = gp_kernel.alpha * exp( -(0.5 / gp_kernel.ell2) * dist2 );

end

