function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
for j = 1: m
    temp(j) = 0;
    for i = 1: size(theta)
        temp(j) = temp(j) + theta(i) * X(j, i);
    end
end
for j = 1: m
    h(j) = sigmoid(temp(j));
end
for j = 1 : m
    J = J + y(j) * log(h(j)) + (1 - y(j))*log(1-h(j));
end
J = -J / m;

for i = 1 : size(theta)
    grad(i) = 0;
    for j = 1 : m
        grad(i) = grad(i) + (h(j) - y(j)) * X(j,i);
    end
    grad(i) = grad(i) / m;
end

% =============================================================

end
