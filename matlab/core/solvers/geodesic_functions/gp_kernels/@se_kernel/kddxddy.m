function [ Kddxddy] = kddxddy( gp_kernel, x, y )

    Kddxddy = (delta(gp_kernel, x, y).^4 - 6 * (delta(gp_kernel, x, y).^2) / gp_kernel.ell2 + 3 / (gp_kernel.ell2^2)) .* kxy(gp_kernel, x, y);

end