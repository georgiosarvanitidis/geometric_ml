function [ M, dMdz, Jacobian] = metric_tensor( manifold, z )
% This function implements the analytic expected Riemannian metric
% from a deep generator (VAE). The metric is expressed in the latent space.
%
% See for details:
% "Latent Space Oddity: on the Curvature of Deep Generative Models",
%   G. Arvanitidis, L. K. Hansen, S. Hauberg,
%   International Conference on Learning Representations (ICLR) 2018.
%
% Input:
%   z - The latent point d x N.
%
% Output:
%   M        - The metric tensor for N points Nx(DxD).
%   dMdz     - The metric tensor derivatives Nx(DxD)xD. For each point N
%              the M(n,:,:,j) is the dM(z)/dz_j (derivative wrt dz_j).
%   Jacobian - The Jacobian matrix Nx2xDxd. For each N the J(n,j,:,:) is
%              the Jacobian of the mean (j=1) and sigma (j=2) respectively.
%
% Author: Georgios Arvanitidis
    
    %% Get the parameters of the network
    W0 = manifold.W0; b0 = manifold.b0;
    W1 = manifold.W1; b1 = manifold.b1;
    W2 = manifold.W2; b2 = manifold.b2;

    Wrbf   = manifold.Wrbf;
    C      = manifold.C;
    gammas = manifold.gammas;
    zeta = manifold.zeta;
    
    %% Define the activation functions of the deep net for the mean function
%     f2 = @(x)linearFun(x); df2 = @(x)dlinearFun(x); ddf2 = @(x)ddlinearFun(x);
%     f2 = @(x)sigmoid(x); df2 = @(x)dsigmoid(x); ddf2 = @(x)ddsigmoid(x);
%     f1 = @(x)tanh(x); df1 = @(x)dtanh(x); ddf1 = @(x)ddtanh(x);
%     f0 = @(x)tanh(x); df0 = @(x)dtanh(x); ddf0 = @(x)ddtanh(x);
    
    f2 = @(x)tanh(x); df2 = @(x)dtanh(x); ddf2 = @(x)ddtanh(x);
    f1 = @(x)softplus(x); df1 = @(x)dsoftplus(x); ddf1 = @(x)ddsoftplus(x);
    f0 = @(x)softplus(x); df0 = @(x)dsoftplus(x); ddf0 = @(x)ddsoftplus(x);

    %% Get the parameters of the data
    D = size(Wrbf,1);   % The dimension of the feature space
    [d, N] = size(z);   % The dimension of the latent space d

    %% Predefine the output matrices
    M = zeros(N, d, d);
    
    %% Return the derivative of the metric tensor?
    if (nargout > 1)
        dMdz = zeros(N, d, d, d);
    end
    
    %% Return the Jacobian matrix?
    if (nargout > 2)
        Jacobian = zeros(N, 2, D, d);
    end
    
    %% For speed-up do these computations
    W0Z0b0 = W0 * z + b0;
    f0W0Z0b0 = f0(W0Z0b0);
    W1f0W0Z0b0b1 = W1 * f0W0Z0b0 + b1;
    W2f1W1f0W0Z0b0b1b2 = W2 * f1(W1f0W0Z0b0b1) + b2;
    
    %% Compute the metric tensor and the derivative of it for every n
    for n = 1:N
        %% The mu network
        % The derivative of the mu network, Jacobian D x d
        dmudz = bsxfun(@times, df2(W2f1W1f0W0Z0b0b1b2(:, n)), W2 ) ...
              * (bsxfun(@times, df1(W1f0W0Z0b0b1(:, n)), W1) ... 
              * bsxfun(@times, df0(W0Z0b0(:, n)), W0));

        %% The sigma network
        % The derivative of the sigma network, Jacobian D x d
        dsigmadz = -0.5 * bsxfun(@rdivide, drbf(z(:, n), C, gammas, Wrbf), ...
            sqrt(bsxfun(@power, rbf(z(:, n), C, gammas, Wrbf, zeta), 3)));

        %% Return the Jacobian if it is needed
        if (nargout > 2)
            Jacobian(n, 1, :, :) = dmudz;
            Jacobian(n, 2, :, :) = dsigmadz;
        end % if
        
        %% The metric tensor expectation d x d
        M(n, :, :) = dmudz' * dmudz + dsigmadz' * dsigmadz;

        %% Do we need the derivative of the metric?
        if( nargout > 1)            
            for dd = 1:d
                
                dJmudzd = bsxfun(@times, bsxfun(@times, ddf2(W2f1W1f0W0Z0b0b1b2(:, n)), W2) ...
                        * bsxfun(@times, df1(W1f0W0Z0b0b1(:, n)), W1) ...
                        * bsxfun(@times, df0(W0Z0b0(:, n)), W0(:, dd)), W2) ...
                        * (bsxfun(@times, df1(W1f0W0Z0b0b1(:, n)), W1) ...
                        * bsxfun(@times, df0(W0Z0b0(:, n)), W0)) ...
                        + ...
                          bsxfun(@times, df2(W2f1W1f0W0Z0b0b1b2(:, n)), W2) ...
                        * (bsxfun(@times, bsxfun(@times, ddf1(W1f0W0Z0b0b1(:, n)), W1) ...
                        * bsxfun(@times, df0(W0Z0b0(:, n)), W0(:,dd)), W1) ...
                        * bsxfun(@times, df0(W0Z0b0(:, n)), W0)) ...
                        + ...
                          bsxfun(@times, df2(W2f1W1f0W0Z0b0b1b2(:, n)), W2) ...
                        * (bsxfun(@times, df1(W1f0W0Z0b0b1(:, n)), W1) ...
                        * bsxfun(@times, bsxfun(@times, ddf0(W0Z0b0(:, n)), W0(:, dd)), W0));
                        
                
                dJsigmadzd = ddrbf(Wrbf, C, gammas, z(:, n), dd, zeta);

                dMdz(n, :, :, dd) = dJmudzd' * dmudz + dmudz' * dJmudzd ...
                                 + dJsigmadzd' * dsigmadz + dsigmadz' * dJsigmadzd;
            end % for
        end % if
    end % for
end % function


%% The derivative of the Jacobian of the variance network
function val = ddrbf(Wrbf, C, gammas, z, d, zeta)  
    temp = zeros(size(C));
    temp(:, d) = 1;
    
    dist   = sum(z.^2, 1) + sum(C.^2, 2) - 2 * C * z;
    rbfVal = exp(-gammas .* dist);
    output = Wrbf * exp(-gammas .* dist) + zeta;  
    
    val = -0.5 * bsxfun(@times, bsxfun(@times, -1.5 * bsxfun(@rdivide, 1, sqrt(bsxfun(@power, output, 5))), ...
          Wrbf * (-2 * bsxfun(@times, gammas, bsxfun(@times, (z(d) - C(:,d)), rbfVal)))), ...
          -2 * Wrbf * bsxfun(@times, bsxfun(@times, gammas, (z' - C)), rbfVal)) ...
        + ...
          -0.5 * bsxfun(@times, bsxfun(@rdivide, 1, sqrt(bsxfun(@power, output, 3))), ...
          (Wrbf * ( -2 * bsxfun(@times, gammas, (bsxfun(@times, temp, rbfVal) - ...
          2 * bsxfun(@times, bsxfun(@times, (z(d) - C(:,d)), gammas), ...
          bsxfun(@times, (z' - C), rbfVal)))))));
end

function val = drbf(z, C, gammas, Wrbf)
    dist   = sum(z.^2, 1) + sum(C.^2, 2) - 2 * C * z;
    rbfVal = exp(-bsxfun(@times, gammas, dist));
    val    = -2 * Wrbf * bsxfun(@times, bsxfun(@times, gammas, (z' - C)), rbfVal);
end

function val = rbf(z, C, gammas, Wrbf, beta)
    dist   = sum(z.^2, 1) + sum(C.^2, 2) - 2 * C * z;
    val  = Wrbf * exp(-bsxfun(@times, gammas, dist)) + beta;
end

%% The activation functions and their derivatives
function val = linearFun(z)
    val = z;
end

function val = dlinearFun(z)
    val = ones(size(z));
end

function val = ddlinearFun(z)
    val = zeros(size(z));
end

function val = softplus(z)
    val = log(1 + exp(z));
end

function val = dsoftplus(z)
    val = bsxfun(@rdivide, 1, 1 + exp(-z));
end

function val = ddsoftplus(z)
    val = bsxfun(@rdivide, exp(z), bsxfun(@power, 1 + exp(z),2));
end

function val = relu(z)
    val = bsxfun(@max, z, 0);
end

function val = drelu(z)
    val = bsxfun(@max, z, 0);
    val = double(bsxfun(@gt, val, 0));
end

function val = ddrelu(z)
    val = zeros(size(z));
end

function val = sigmoid(z)
    val = sigmf(z, [1, 0]);
end

function val = dsigmoid(z)
    val = bsxfun(@times, sigmoid(z), 1 - sigmoid(z));
end

function val = ddsigmoid(z)
    val = bsxfun(@times, dsigmoid(z), 1 - 2 * sigmoid(z));
end

function val = dtanh(z)
    val = (1 - bsxfun(@power, tanh(z), 2));
end

function val = ddtanh(z)
    val = -2 * bsxfun(@times, tanh(z), 1 - bsxfun(@power, tanh(z), 2));
end