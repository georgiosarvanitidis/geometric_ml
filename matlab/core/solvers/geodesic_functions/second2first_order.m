function y = second2first_order(manifold, state)
  % Dimensions:
  % state: (2D)xN
  % y: (2D)xN

  D = size(state, 1) / 2;
  c = state(1:D, :);
  cm = state((D+1):end, :);
  cmm = geodesic_system(manifold, c, cm);
  y = [cm; cmm];
end % function
