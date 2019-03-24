function [ Kdxdy ] = kdxdy( gp_kernel, x, y )

    Kdxdy = (1 / gp_kernel.ell2 - delta(gp_kernel, x, y).^2) .* kxy(gp_kernel, x, y);

end

