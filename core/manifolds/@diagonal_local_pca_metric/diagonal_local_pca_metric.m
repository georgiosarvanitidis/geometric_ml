function manifold = diagonal_local_pca_metric (data, sigma, rho)
  
  %% Create manfold object
  manifold = struct ();
  
  manifold.X = data; % RxD
  manifold.sigma2 = sigma^2; % scalar
  manifold.rho = rho; % scalar
  manifold.dimension = size(data, 2); % scalar
 
  manifold = class(manifold, 'diagonal_local_pca_metric');
end % function