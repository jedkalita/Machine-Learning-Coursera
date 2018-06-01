function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%X = [ones(m, 1) X];
h_theta = X * theta;
tmp1 = h_theta - y;
tmp2 = tmp1 .^ 2;
sum_1 = sum(tmp2);
J = (1/ (2 * m)) * sum_1;
theta_sq = theta .^ 2;
sum_theta_sq = sum(theta_sq);
sum_theta_sq = sum_theta_sq - theta_sq(1);
t = (lambda / (2 * m));
sum_theta_sq = t * sum_theta_sq;

J = J + sum_theta_sq;


%now calculate the gradient
for i = 1:m
    grad(1, 1) = grad(1, 1) + (tmp1(i, 1) * X(i, 1));
end
grad(1, 1) = grad(1, 1) / m;

t2 = (lambda / m);
for j = 2:rows(grad)
    for i = 1:m
        grad(j, 1) = grad(j, 1) + (tmp1(i, 1) * X(i, j));
    end
    grad(j, 1) = (grad(j, 1) / m) + (t2 * theta(j, 1)) ;
end






% =========================================================================

grad = grad(:);

end
