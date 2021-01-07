function [ Kxdy ] = kxdy( gp_kernel, x, y )

    Kxdy = - kdxy(gp_kernel, x, y);
    
end

