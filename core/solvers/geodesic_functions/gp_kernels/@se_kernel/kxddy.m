function [ Kxddy ] = kxddy( gp_kernel, x, y )

    Kxddy = -kdxdy(gp_kernel, x, y);

end

