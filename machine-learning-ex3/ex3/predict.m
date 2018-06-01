function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m, 1) X];
z1 = Theta1 * X';
a2 = sigmoid(z1);
a2_tmp = a2';
m2 = size(a2_tmp, 1);
a2_tmp = [ones(m2, 1) a2_tmp];
z2 = Theta2 * a2_tmp';
a3 = sigmoid(z2);
h_theta = a3';
for i = 1:m
    [a p(i,1)] = max(h_theta(i, :));
end








% =========================================================================


end
