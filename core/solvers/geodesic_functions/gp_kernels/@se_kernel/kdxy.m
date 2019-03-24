function [ Kdxy ] = kdxy( gp_kernel, x, y )

    Kdxy = - delta(gp_kernel, x, y) .* kxy(gp_kernel, x, y);
    
end

