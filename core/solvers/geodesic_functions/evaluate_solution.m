function [c, dc] = evaluate_solution (solution, t, t_scale)
  cdc = deval (solution, t*t_scale); % 2DxT
  D  = size (cdc, 1) / 2;
  c  = cdc (1:D, :).'; % TxD
  dc = cdc ((D+1):end, :).' * t_scale; % TxD
  c = c.'; dc = dc.';
end % function
