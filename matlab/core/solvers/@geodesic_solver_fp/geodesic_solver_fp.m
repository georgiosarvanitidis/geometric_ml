function solver = geodesic_solver_fp(D, options)
% This function constructs an object from the specified solver.
%   
% Options:
%   N        - number of mesh points [0,1] including boundaries.
%   tol      - the tolerance of the algorithm.
%   ell      - the length scale.
%   max_iter - the maximum iterations.
%   Sdata    - the amplitude of the kernel.
%   kernel   - the kernel for the GP.
%   sigma    - the std of the noise for the GP parameters.
    
    % If the dimensionality is not given the solver does not work.
    if(nargin == 0)
        error('Solver dimensionality has not been set...')
    end
    
    % If the default parameters to be used.
    if(nargin == 1)
        options = [];
    end
    
    % Initialize the solver options
    solver.options = struct();
    
    if( isfield(options, 'N') )
        solver.N = options.N;
    else
        solver.N = 10;
    end
    
    if( isfield(options, 'tol') )
        solver.tol = options.tol;
    else
        solver.tol = 1e-1;
    end
    
    % The mesh size
    T = linspace(0, 1, solver.N);
    solver.T = T;
    
    if( isfield(options, 'ell') )
        solver.ell = options.ell;
    else
        solver.ell = sqrt(0.5*(T(2) - T(1)));
    end
    
    if( isfield(options, 'max_iter') )
        solver.max_iter = options.max_iter;
    else
        solver.max_iter = 1000;
    end
    
    if( isfield(options, 'Sdata') )
        solver.Sdata = options.Sdata;
    else
        warning('Kernel amplitude has not been specified!');
        solver.Sdata = eye(D);
    end
    
    if( isfield(options, 'kernel_type') )
        error('Kernel type is not given... ');
    else
        warning('Kernel type has not been specified (default: Squared Exponential).');
        solver.gp_kernel = se_kernel(solver.ell, 1); % ell, alpha (amplitude=1)
    end
    
    % This is the fixed noise of the parameters that we learn during training.
    if( isfield(options, 'sigma') )
        solver.sigma2 = options.sigma^2;
    else
        solver.sigma2 = (1e-4)^2;
    end
    
    % Initialize the object
    solver = class(solver, 'geodesic_solver_fp');

end