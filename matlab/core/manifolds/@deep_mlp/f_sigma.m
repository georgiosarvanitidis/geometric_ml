function [ val ] = f_sigma( manifold, z )

    val = bsxfun(@rdivide, 1, sqrt(rbf(z, manifold)));

end

% The output of the rbf function
function val = rbf(z, manifold)
    dist   = sum(z.^2, 1) + sum(manifold.C.^2, 2) - 2 * manifold.C * z;
    val  = manifold.Wrbf * exp(-bsxfun(@times, manifold.gammas, dist)) + manifold.zeta;
end