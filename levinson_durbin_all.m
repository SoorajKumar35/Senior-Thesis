function h = levinson_durbin_all(T, y)
% inputs: T - one row of the Toeplitz matrix, y - The recording
% outputs: h - the impulse response

N = length(T);
f = zeros(N);
b = zeros(N);

% First, for n = 1
f(1) = 1/T(1,1);
b(N) = 1/T(1,1);

% Second, for n = 2:N
for i=2:N
    
end
end