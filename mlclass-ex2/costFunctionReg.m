function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

reg = 0;
for i = 2: size(theta)
    reg = reg + theta(i) ^ 2;
end
reg = reg * lambda * 0.5 /m;
J = J + reg;

for i = 1 : size(theta)
    grad(i) = 0;
    for j = 1 : m
        grad(i) = grad(i) + (h(j) - y(j)) * X(j,i);
    end
    grad(i) = grad(i) / m;
end
for i = 2 : size(theta)
    grad(i) = grad(i) + lambda * theta(i) / m;
end


% =============================================================

end
