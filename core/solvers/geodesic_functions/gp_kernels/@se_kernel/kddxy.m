function [ Kddxy ] = kddxy( gp_kernel, x, y )

    Kddxy = -kdxdy(gp_kernel, x, y);

end

