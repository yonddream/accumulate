function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
    % 梯度下降函数写成代码就是下面的方式
    tmp = theta - alpha / m * X' * (X * theta - y);
    % fprintf(' %f \n', tmp);
    % 上下2种写法结果都一样，上面是下面的向量化演算结果。但是只有在线性回归中才相等。下面有论证过程 
    % http://stackoverflow.com/questions/10479353/gradient-descent-seems-to-fail/42330833#42330833 
    % 突然顿悟 这是2次Vectorization的结果 所以是上面的结果 
    %       Vectorization example.
    % Unvectorized implementation
    % double prediction = 0.0; 
    % for (int j = 0; j < n; j++)
    %     prediction += theta[j] * x[y];

    % Vectorized implementation
    % double prediction = theta.transpose() * x;
    

    % theta(1...n) 需要一起变动所以需要一个临时变量来存储
    hold = theta;
    for j = 1:length( theta ) 
        theta(j) =  hold(j) - ( alpha * sum(( X * hold - y ) .* X(:, j))) / m;
    end
    % fprintf(' %f \n', theta);
    % fprintf('\n');
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
