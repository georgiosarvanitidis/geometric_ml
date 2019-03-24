function [Curves, Logs, Lens, Fails, Solutions, time_elapsed] = compute_all_geodesics (solver, manifold, mu, data)
% This function computes all the shortest paths between the point mu and the data.    
    
    [N, D] = size(data);
    
    % Initialyze the output
    Curves = cell(N, 1);
    Lens = NaN(N, 1);
    Logs = NaN(N, D);
    Fails = false(N, 1);
    Solutions = cell(N, 1);
    
    t = tic();
    for n = 1:N
        fprintf ('Computing geodesic %d / %d\n', n, N);
        [curve, logmap, len, failed, solution] = compute_geodesic(solver, manifold, mu, data(n, :));
        Curves{n} = curve;
        Lens(n) = len;
        Logs(n, :) = logmap;
        Fails(n) = failed;
        Solutions{n} = solution;
    end % for
    
    time_elapsed = toc(t);
end % function
