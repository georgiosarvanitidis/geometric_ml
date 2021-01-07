function [ d ] = delta( gp_kernel, x, y )

    d = (x(:) - y(:).') ./ gp_kernel.ell2;

end

