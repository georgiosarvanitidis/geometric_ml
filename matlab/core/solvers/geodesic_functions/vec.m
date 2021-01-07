% VEC - vectorizes an input matrix
%
% For a MxN matrix, returns a column vector of size MN by stacking columns
% on top of each other.
%
% Inputs: A - MxN matrix to be put in vector form
%
% Returns: v - column vector of size MN containing the elements of A
function v = vec (A)
  v = A(:);
end % function