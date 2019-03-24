function manifold = deep_mlp(muNet, sigmaNet)
    %% Create manifold object
    manifold = struct ();
    
    % The weights of the mu network
    manifold.W2 = muNet.W2; manifold.b2 = muNet.b2; % D  x L2
    manifold.W1 = muNet.W1; manifold.b1 = muNet.b1; % L2 x L1
    manifold.W0 = muNet.W0; manifold.b0 = muNet.b0; % L1 x d

    % The parameters of the variance network
    manifold.Wrbf   = sigmaNet.Wrbf; % D x L
    manifold.gammas = sigmaNet.gammas; % L x 1
    manifold.C      = sigmaNet.centers; % L x d
    manifold.zeta   = sigmaNet.zeta;    % scalar
    
    manifold.dimension = size(manifold.W0, 2); % scalar
 
    manifold = class(manifold, 'deep_mlp');
end % function