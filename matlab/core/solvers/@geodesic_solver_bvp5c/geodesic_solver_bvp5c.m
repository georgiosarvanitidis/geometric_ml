function solver = geodesic_solver_bvp5c(options)
% This function constructs an object from the specified solver.

    % Initialize the options of the solver
    solver.options = struct();
    
    if(isfield(options, 'NMax') )
        solver.NMax = options.NMax;
    else
        solver.NMax = 1000;
    end % if
        
    if(isfield(options, 'tol') )
        solver.tol = options.tol;
    else
        solver.tol = 1e-1;
    end % if
    
    % Initialize the class
    solver = class(solver, 'geodesic_solver_bvp5c');
end