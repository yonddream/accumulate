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
% https://www.coursera.org/learn/machine-learning/supplement/0hpMl/simplified-cost-function-and-gradient-descent
% 计算h(θ)
h = sigmoid(X * theta);
J = (-y' * log(h) - (1 - y)' * log(1 - h)) / m;

% 用的高级算法不是梯度下降 所以只需要计算每个
for j = 1: length(grad)
	grad(j) = sum((h - y) .* X(:, j)) / m;
end

% 上下2个结果是相等的下面是向量化
grad = (X' * (h - y)) / m;



% =============================================================

end
