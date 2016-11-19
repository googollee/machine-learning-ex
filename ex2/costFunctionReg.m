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

p1 = - y' * (log(sigmoid(X * theta)));
p2 = - (1-y)' * (log(1 - sigmoid(X * theta)));
reg_theta = theta;
reg_theta(1) = 0;
p3 = lambda * (reg_theta' * reg_theta) / 2 / m;
J = (p1 + p2) ./ m + p3;

p4 = lambda * reg_theta ./ m;
grad = X' * (sigmoid(X * theta) - y) ./ m + p4;


% =============================================================

end
