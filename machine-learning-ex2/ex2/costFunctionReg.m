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

%cost calculation
z = X * theta;
h_theta = sigmoid(z);
for i = 1:m
    J = (J + ((-1 * y(i, 1) * log(h_theta(i, 1))) - ((1 - y(i, 1)) * log(1 - h_theta(i, 1)))));
end
J = (J / m);
sum2 = 0;
for i = 2:rows(theta)
    sum2 = sum2 + (theta(i, 1)^2);
end
sum2 = (sum2 / (2 * m)) * lambda;
J = J + sum2;

%now calculate the gradient
for i = 1:m
    grad(1, 1) = grad(1, 1) + ((h_theta(i) - y(i)) * X(i, 1));
end
grad(1, 1) = grad(1, 1) / m;

for j = 2:rows(grad)
    for i = 1:m
        grad(j, 1) = grad(j, 1) + ((h_theta(i) - y(i)) * X(i, j));
    end
    grad(j, 1) = (grad(j, 1) / m) + ((lambda / m) * theta(j, 1)) ;
end




% =============================================================

end
