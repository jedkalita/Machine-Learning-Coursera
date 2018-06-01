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

% k = num_labels;
%sum = 0;
% for i = 1:m
%    y_i = zeros(k, 1);
%    for l = 1:k
%        if (l == y(i, 1))
%            y_i(l) = 1;
%            break
%    end
%    h = h_theta(i, :)';
%    for j = 1:k
%        t1 = (-1 * y_i(j, 1));
%        t2 = log(h(j, 1));
%        t12 = t1 * t2;
%        t3 = (1 - y_i(j, 1));
%        t4 = log(1 - h(j, 1));
%        t34 = t3 * t4;
%        t1234 = t12 - t34;
%        J = J + t1234;
              
%    end
% end
% J = J / m;
y_matrix = eye(num_labels)(y, :); %5000x10
a1 = [ones(m, 1) X]; %5000x401
z2 = Theta1 * a1'; %25x5000
a2 = sigmoid(z2); %25x5000
a2_tmp = a2'; %5000x25
m2 = size(a2_tmp, 1);
a2_tmp = [ones(m2, 1) a2_tmp]; %5000x26
z3 = Theta2 * a2_tmp'; %10x5000
a3 = sigmoid(z3); %10x5000
a3 = a3'; %5000x10

minus_y_matrix = -1 * y_matrix;
log_a3 = log(a3);
y2_matrix = 1 - y_matrix;
minus_a3 = 1 - a3;
log_minus_a3 = log(minus_a3);
mat1 = minus_y_matrix .* log_a3;
mat2 = y2_matrix .* log_minus_a3;
mat_fin = mat1 - mat2;
mat_fin_fin = mat_fin';
J = sum(sum(mat_fin_fin));
J = (J / m);

m_1_1 = size(Theta1, 1);
m_1_2 = size(Theta1, 2);
m_2_1 = size(Theta2, 1);
m_2_2 = size(Theta2, 2);
theta1_sq = Theta1(1:m_1_1, 2: m_1_2) .^ 2;
theta2_sq = Theta2(1:m_2_1, 2: m_2_2) .^ 2;
s1 = sum(sum(theta1_sq));
s2 = sum(sum(theta2_sq));
sum_theta_sq = s1 + s2;
sum_theta_sq = (sum_theta_sq * (lambda / (2 * m)));
J = J + sum_theta_sq;
% -------------------------------------------------------------

% =========================================================================

d3 = a3 - y_matrix; %5000x10
z_2 = z2'; %5000x25
d2_tmp = d3 * Theta2(:, 2:end); %5000x25
sig_grad_z2 = sigmoidGradient(z_2); %5000x25
d2 = d2_tmp .* sig_grad_z2; %5000x25
delta1 = d2' * a1; %25x5000 * 5000x401 = 25x401
delta2 = d3' * a2_tmp; %10x5000 * 5000x26 = 10x26
Theta1_grad = (1/m) * delta1;
Theta2_grad = (1/m) * delta2;


add_mat1 = (lambda/m) * Theta1;
add_mat2 = (lambda/m) * Theta2;
add_mat1(:, 1) = 0;
add_mat2(:, 1) = 0;
Theta1_grad = Theta1_grad + add_mat1;
Theta2_grad = Theta2_grad + add_mat2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

%important tutorial for back-prop - https://www.coursera.org/learn/machine-learning/discussions/%i2u9QezvEeSQaSIACtiO2Q/replies/XpcX6-0PEeS0tyIAC9RBcw
%important tutorial for forward prop - https://www.coursera.org/learn/machine-learning/discussions/%QFnrpQckEeWv5yIAC00Eog
end
