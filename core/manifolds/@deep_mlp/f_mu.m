function [ val ] = f_mu( manifold, z )

    %% Define the decoding functions
    f2 = @(x)tanh(x);
    f1 = @(x)softplus(x);
    f0 = @(x)softplus(x);
    
    layer0 = f0(manifold.W0 * z + manifold.b0);
    layer1 = f1(manifold.W1 * layer0 + manifold.b1);
    layer2 = f2(manifold.W2 * layer1 + manifold.b2);
    val = layer2;
    
end

%% The activation functions
function val = sigmoid(z)
    val = sigmf(z, [1, 0]);
end

function val = softplus(z)
    val = log(1 + exp(z));
end
