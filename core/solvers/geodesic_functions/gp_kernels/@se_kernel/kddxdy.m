function [ Kddxdy ] = kddxdy( gp_kernel, x, y )

    Kddxdy = ( - 3 * delta(gp_kernel, x, y) / gp_kernel.ell2 + delta(gp_kernel, x, y).^3) .* kxy(gp_kernel, x, y);

end

