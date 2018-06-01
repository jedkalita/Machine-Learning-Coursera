function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
theta_X = X * Theta';
theta_X_minusY = theta_X - Y;
theta_X_minusY_sq = theta_X_minusY .^ 2;
theta_X_minusY_sq_R = theta_X_minusY_sq .* R;
J = sum(sum(theta_X_minusY_sq_R));

J = (J / 2);

theta_X_minusY_R = theta_X_minusY .* R;
X_grad = theta_X_minusY_R * Theta;

theta_X_minusY_R_transpose = theta_X_minusY_R';
Theta_grad = theta_X_minusY_R_transpose * X;



                   

sum_theta_sq = sum(sum(Theta .^ 2));
sum_X_sq = sum(sum(X .^ 2));

sum_theta_sq = (sum_theta_sq * lambda) / 2;
sum_X_sq = (sum_X_sq * lambda) / 2;

J = J + sum_theta_sq + sum_X_sq;

%X_lambda = X * lambda;
%Theta_lambda = Theta * lambda;
%X_grad = X_grad + X_lambda;
%Theta_grad = Theta_grad + Theta+lambda;

for i = 1:num_movies
    X_grad(i, :) = X_grad(i, :) + (lambda * X(i, :));
end

for j = 1:num_users
    Theta_grad(j, :) = Theta_grad(j, :) + (lambda * Theta(j, :));
end






% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
