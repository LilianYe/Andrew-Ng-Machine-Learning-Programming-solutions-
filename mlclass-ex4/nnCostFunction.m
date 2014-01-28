function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a1 = [ones(size(X,1),1),X]; %5000 * 201
z2 = a1 * Theta1'; %5000 * 25
a2 = sigmoid(z2); %5000 * 25
a2 = [ones(size(a2,1),1),a2]; % 5000 * 26
a3 = sigmoid(a2 * Theta2'); % 5000 * 10
m = size(X,1); %5000
K = size(a3, 2); %10
y1 = zeros(m, K);
one = ones(m, K);
for k = 1:K
    y1(:,k) = y == k;
end
%y1(:,K) = y == 0;
J = sum(sum(-y1.*log(a3)-(one-y1).*log(one-a3)));
J = J / m;
theta1 = Theta1(:,2:size(Theta1,2));%25*400
theta2 = Theta2(:,2:size(Theta2,2)); %10*25
J = J + lambda / 2 / m * (sum(sum(theta1.^2))+ sum(sum(theta2.^2)));
% -------------------------------------------------------------

% =========================================================================
error3 = a3 - y1; %5000 * 10
error2 = error3*theta2.*sigmoidGradient(z2); %5000 * 25
%error1 = error2*theta1.*sigmoidGradient(z1); %5000 * 400
% Unroll gradient
Theta1_grad = Theta1_grad + error2'*a1; %25*401
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad + error3'*a2; %10*26
Theta2_grad = Theta2_grad / m;
theta1 = [zeros(size(theta1,1),1),theta1];
theta2 = [zeros(size(theta2,1),1),theta2];
Theta1_grad = Theta1_grad + lambda / m * theta1;
Theta2_grad = Theta2_grad + lambda / m * theta2;
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
