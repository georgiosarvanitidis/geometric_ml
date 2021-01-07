function [ Kdxddy ] = kdxddy( gp_kernel, x, y )

    Kdxddy = - kddxdy(gp_kernel, x, y);
    
end

