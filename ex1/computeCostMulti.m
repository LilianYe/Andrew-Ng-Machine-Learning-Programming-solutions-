function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
h = 0;
p = zeros(m, 1);
for i=1:m  
    for j = 1 : length(theta)
        p(i) = p(i) + theta(j) * X(i, j);
    end
 end
for i = 1 : m
    h = h + (p(i) - y(i))^2;
end
J = 0.5 * h / m;

% =========================================================================

end
