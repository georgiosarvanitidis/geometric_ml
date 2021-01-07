function [curve, logmap, len, failed, solution] = compute_geodesic( solver, manifold, c0, c1, solution)
% This function computes the shortest path between two given points on a
% Riemannian manifold. Details for the fixed point method that is utilized
% can be found in:
% "Fast and Robust Shortest Paths on Manifolds Learned from Data",
%   G. Arvanitidis, S. Hauberg, P. Hennig, M. Schober,
%   International Conference on Artificial Intelligence and Statistics
%   (AISTATS) 2019.
%
% Input:
%   solver   - the object from the sovler class.
%   manifold - An object of any manifold class.
%   c0, c1   - The given boundary points Dx1.
%   solution - A struct with the solution of the shortest path.
%
% Output:
%   curve    - A parametric function c(t):[0,1] -> R^D.
%   logmap   - The logarithmic map Dx1.
%   len      - The length of the geodesic.
%   failed   - A boolean, if solver failed this value is true.
%   solution - The struct containing the details of the solution.
%
% Author: Georgios Arvanitidis

    %% Get boundary points and ensure column vectors
    c0 = c0(:);
    c1 = c1(:);
    D = numel(c0);
    
    %% Parameters
    t_min = 0; t_max = 1;   % the boundaries of the interval to integrate [0,1]
    N = solver.N;   % number of points to estimate
    max_iter = solver.max_iter; % maximum allowed iterations
    ell = solver.ell;   % the length-scale of the kernel
    T = solver.T'; % N x 1, the discretized time interval [0,1]
    tol = solver.tol; % algorithm tolerance
    gp_kernel = solver.gp_kernel; % the GP kernel
    
    %% The definition of the GP for curve, dcurve, ddcurve, c0, c1
    
    % ----- Estimate the amplitude of the kernel and define ell
    v = c1 - c0; % D x 1
    Veta = ((v' * solver.Sdata * v) * solver.Sdata); % Estimate in Bayesian style the amplitude
    
    % ----- The prior mean functions
    m = @(t) c0' + t * (c1 - c0)'; % T x D
    dm = @(t) ones(size(t, 1), 1) * (c1 - c0)'; % T x D
    ddm = @(t) zeros(size(t, 1), 1) * (c1 - c0)'; % T x D
    
    % ----- The kernel components for the GP ( KernelMatrix = [A, C; C', B + R] )
    I_D = eye(D);   % D x D identity matrix
   
    R = @(sigma2)blkdiag(sigma2{:}, I_D * 1e-10, I_D * 1e-10); % The noise block matrix
    
    % ----- The kernel components    
    Ctrain = [kdxddy(gp_kernel, T, T), kdxy(gp_kernel, T, [t_min; t_max]); ...
              kxddy(gp_kernel, T, T), kxy(gp_kernel, T, [t_min; t_max])]; % 2N x (N+2)
      
    Btrain = [kddxddy(gp_kernel, T, T), kddxy(gp_kernel, T, [t_min; t_max]); ...
              kxddy(gp_kernel, [t_min; t_max], T), kxy(gp_kernel, [t_min; t_max], [t_min; t_max])]; % (N+2)x(N+2)
     
    % ----- This is the vector where we keep the observations
    y_hat = [ddm(T); m([t_min; t_max])]; % Fixed prior mean, (N+2) x D
    y_obs = @(ddc)[ddc; c0(:)'; c1(:)']; % (N+2) x D
    
    % ----- The initial guess for the parameters c'' we want to learn
    if(nargin < 5)
        DDC = ddm(T);  % N x D
    else
        % If the geodesic has been solved previously, we will initialize
        % the current parameters with the ones found before.
        if(~isempty(solution))
            DDC = solution.ddc;
        end
    end
    
    % ----- Initialize the noise of the GP
    sigma2_blkdiag = cell(N, 1);  % N x 1 (every cell DxD)
    for n = 1:N
        sigma2_blkdiag{n} = solver.sigma2 * I_D;
    end
    
    % ----- The posterior mean of the dcurve & curve with size 2*(N*D) x 1
    Btrain_R_inv = (kron(Btrain, Veta) + R(sigma2_blkdiag)); % precompute for speed
    Btrain_R_inv = Btrain_R_inv \ eye(size(Btrain_R_inv));
    kronCtrainVeta = kron(Ctrain, Veta);
    dmu_mu_post = @(t, ddc)vec([dm(t); m(t)]') + kronCtrainVeta * (Btrain_R_inv * vec((y_obs(ddc) - y_hat)'));
    
    %% The main loop to train z(t_n)'' ~= c''(t_n)
    iter = 1;
    tic;
    try
        while(1)

            % ----- Compute current c, dc posterior
            dmu_mu_post_curr = reshape(dmu_mu_post(T, DDC), [D, 2 * N]); % D x 2N
            mu_post_curr = dmu_mu_post_curr(:, N+1:end); % D x N
            dmu_post_curr = dmu_mu_post_curr(:, 1:N);    % D x N

            % ----- Evaluate the c'' = f(c, c') the fixed-point
            DDC_new = geodesic_system(manifold, mu_post_curr, dmu_post_curr)';
            
            % ----- Not significant change in parameters then stop or max iter
            cost_current = (DDC - DDC_new).^2;
%             disp(['Iter: ',num2str(iter),', cost: ', num2str(sum(sum(cost_current)))]);
            condition_1 = all(all((cost_current) < tol));
            condition_2 = (iter > max_iter);
            if( condition_1 || condition_2)
                if( condition_1 ), convergence_cond = 1; end
                if( condition_2 ), convergence_cond = 2; end
                break;
            end % if

            
            % ----- Compute the "gradient"
            grad = DDC - DDC_new;

            % ---- The step-size selection
            alpha = 1;
            for i = 1:3
                % Update the DDC temporary & check if update is good
                % DDC_temp = (1 - alpha) * DDC + alpha * DDC_new;
                DDC_temp = DDC - alpha * grad;

                % Temporary posteriors after updating, check if cost decreased.
                dmu_mu_post_curr_temp = reshape(dmu_mu_post(T, DDC_temp), [D, 2 * N]); % 2N x D
                mu_post_curr_temp = dmu_mu_post_curr_temp(:, N + 1:end); % D x N
                dmu_post_curr_temp = dmu_mu_post_curr_temp(:, 1:N); % D x N

                % If cost is reduced keep the step-size
                cost_temp = (DDC_temp - geodesic_system(manifold, mu_post_curr_temp, dmu_post_curr_temp)').^2;
                if( sum(sum(cost_temp)) <= sum(sum(cost_current)) )
                    break;
                else
                    alpha = alpha * 0.33;
                end
            end % for

             % ----- Update the parameteres
            DDC = DDC - alpha * grad;

            % ----- Increase the iteration by one
            iter = iter + 1;

        end % while
    catch
        convergence_cond = 3;
    end % try
    
    %% Prepare the output.
    if(convergence_cond == 2 || convergence_cond == 3) % if it converges by iterations then failed!
        disp('Geodesic solver (fp) failed!');
        curve = @(t)evaluate_failed_solution(c0, c1, t); % straight line
        len = curve_length(manifold, curve);
        logmap = (c1 - c0);
        logmap = len * logmap ./ norm(logmap);
        failed = true;
        solution = [];
        solution.time_elapsed = toc;
    else
        % This is a GP-posterior mean function, evaluated at new points t
        curve = @(t)curve_eval_gp(t, T, t_min, t_max, Btrain, R(sigma2_blkdiag), Veta, dm, m, DDC, y_hat, y_obs, gp_kernel);
        len = curve_length(manifold, curve);
        [~, logmap] = curve(0); % The dc(0)
        logmap = len * logmap ./ norm(logmap); % Scaled dc(0)
        failed = false;
        
        % Keep solution struct
        solution = [];
        solution.ddc = DDC;
        solution.Sigma2 = sigma2_blkdiag; % The noise block-matrix
        solution.total_iter = iter;
        solution.cost = cost_current;
        solution.ell = ell;
        solution.T = T;
        solution.time_elapsed = toc;
        
        % ----- Return as solution also a spline interpolation
        solution.cdc_spline = @(t)curve_eval_spline(T, mu_post_curr, dmu_post_curr, t);
    end % if
        
end % function

% %% Computes the covariance function of a GP.
% function [diagVal, val] = var_post(A, B, R, C, Veta)
%     val = kron(A, Veta) - kron(C, Veta) * ((kron(B, Veta) + R) \ (kron(C, Veta)'));
%     val = 0.5 * (val + val'); % ensure symmetry!!
%     val = val + eye(size(val)) * 1e-10; % ensure PSD matrix
%     diagVal = diag(val);
% end % function

%% Curve evaluation using spline model
function [c_t, dc_t] = curve_eval_spline(T, c, dc, t)
    D = size(c, 2);
    N = size(t(:), 1);
    c_t = zeros(N, D);
    dc_t = zeros(N, D);
    for d = 1:D
       c_t(:, d) = spline(T, c(:,d), t);
       dc_t(:, d) = spline(T, dc(:,d), t);
    end
end % function

%% Curve evaluation for GP
function [c_t, dc_t] = curve_eval_gp(Ts, T, t_min, t_max, Btrain, R, Veta, dm, m, DDC, y_hat, y_obs, gp_kernel)
    Ts = Ts(:); % the test points t_*, ensure this is a column vector.
    D = size(DDC, 2);
    Ns = numel(Ts); % the number of points in the T_star
    
    % ----- The kernel components
%     Atest = [kdxdy(gp_kernel, Ts, Ts), kdxy(gp_kernel, Ts, Ts); ...
%             kxdy(gp_kernel, Ts, Ts), kxy(gp_kernel, Ts, Ts)]; % 2Ns x 2Ns
     
    Ctest = [kdxddy(gp_kernel, Ts, T), kdxy(gp_kernel, Ts, [t_min; t_max]); ...
            kxddy(gp_kernel, Ts, T), kxy(gp_kernel, Ts, [t_min; t_max])]; % 2Ns x (N+2)
        
    % ----- Evaluate the c, dc posterior for the t_star.
    dmu_mu_Ts = vec([dm(Ts); m(Ts)]') + kron(Ctest, Veta) * ((kron(Btrain, Veta) + R) \ vec((y_obs(DDC) - y_hat)'));
    dmu_mu_Ts = reshape(dmu_mu_Ts, [D, 2 * Ns]); % D x 2Ns
    
    % The values to return
    c_t = dmu_mu_Ts(:, Ns + 1:end); % D x Ns
    dc_t = dmu_mu_Ts(:, 1:Ns);     % D x Ns
    
%     % If we need the variance around the posterior
%     if nargout > 2
%         [~, var_post_curr_Ts] = var_post(Atest, Btrain, R, Ctest, Veta);
%         var_c = var_post_curr_Ts((2 * Ns) + 1:end, (2 * Ns) + 1:end);
%         var_dc = var_post_curr_Ts(1:(2 * Ns), 1:(2 * Ns));
%     end
    
end % function

function [c, dc] = evaluate_failed_solution(p0, p1, t)
  t = t(:); % 1xT
  c = (1 - t) * p0.' + t * p1.'; % TxD
  dc = repmat ((p1 - p0).', numel(t), 1); % TxD
  c = c.'; dc = dc.';
end % function


