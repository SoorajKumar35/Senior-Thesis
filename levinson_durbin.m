function x = levinson_durbin(b, r)
% Inputs: b - output of a linear system, T - input toeplitz mat of a linear
% system
%Outputs: x - coefficients of linear system

n = length(b);

y(2) = -r(2);
x(1) = b(1);
beta = 1;
alpha = -r(2);

for k=2:n
   beta = (1 - (alpha)^2)*beta;
   mu = (b(k-1+1) - dot(r(2:k),x(k-1:-1:1)))/beta;
   v(1:k-1) = (x(1:k-1) + mu*y(k:-1:2));
   x(1:k-1) = v(1:k-1);
   x(k - 1 + 1) = mu;
   if k < n
      alpha = -(r(k+1) + dot(r(2:k),y(k:-1:2)))/beta;
      z(2:k) = y(2:k) + alpha*y(k:-1:2);
      y(2:k) = z(2:k);
      y(k+1) = alpha;
   end
end
end