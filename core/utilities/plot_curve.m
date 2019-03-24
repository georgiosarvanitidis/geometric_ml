function h = plot_curve(curve, varargin)
% PLOT_CURVE   Plot a curve given by a function handle.
%    PLOT_CURVE(c) plot the curve c in the parameter range [0, 1].
%
%    PLOT_CURVE(c, a, b) plot the curve c in the parameter range [a, b].
%
%    PLOT_CURVE(c, ...) as above, but further input arguments are passed to
%    the plot or plot3 command.
%
%    See also PLOT, PLOT3

  %% Check input
  if (nargin < 1)
    error('plot_curve: not enough input arguments');
  end % if
  
  if (numel(varargin) > 1 && isscalar(varargin{1}) && isscalar(varargin{2}))
    t0 = varargin{1};
    t1 = varargin{2};
    plot_opts = varargin(3:end);
  else
    t0 = 0;
    t1 = 1;
    plot_opts = varargin;
  end % if
  
  %% Discretise curve
  if (isa(curve, 'function_handle') || isa(curve, 'gaussian_process') || ...
      isa(curve, 'mockup'))
    N = 100;
    t = linspace(t0, t1, N);
    discrete_curve = curve(t); % DxN
  else
    discrete_curve = curve;
  end % if
  D = size(discrete_curve, 1);
  
  %% Plot the curve
  if (D == 2)
    hh = plot(discrete_curve(1, :), discrete_curve(2, :), plot_opts{:});
  elseif (D == 3)
    hh = plot3(discrete_curve(1, :), discrete_curve(2, :), discrete_curve(3, :), plot_opts{:});
  elseif (D == 1)
    hh = plot(linspace(t0, t1, size(discrete_curve, 2)), discrete_curve, plot_opts{:});
  else
    error('plot_curve: unable to plot curves in %d dimensions', D);
  end % if
  
  %% Does the use want the plotting handle?
  if (nargout > 0)
    h = hh;
  end % if    
end % function