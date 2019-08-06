function P = sooraj_micarray_srp_phat(x,angles,fs)
% Localizes a broadband sound using the Respeaker Array
% x has dimension [N x 8]
% angles (in radians) has dimension [1 x num_angles]
% Output P is the phase-transformed "power" at each angle in angles

C = 343;
% MIC_POS = 0.032*[0 cos((0:60:300)*pi/180); 0 sin((0:60:300)*pi/180); zeros(1,7)]; % [3 x 7]
MIC_POS = [0:0.055:(7*0.055); zeros(1,8); zeros(1,8)];
% Ensure angles is a row vector
if iscolumn(angles), angles = angles.'; end

omega = (0:(size(x,1)-1))/size(x,1)*2*pi; % [1 x N]
delays = -1*MIC_POS'*[0.9252*cos(angles); 0.9252*sin(angles); -0.3794*ones(size(angles))]/C*fs; % [7 x num_angles]

%x = x(:,1:7); % Throw away mysterious eighth channel

% Phase transform
X = fft(x); % N x 7
X = X./abs(X);

% Frequency domain beamforming
P = zeros(1,length(angles));
for a = 1:length(angles)
    W = exp(1j*delays(:,a)*omega).'; % [N x 7]
    Y = sum(W.*X,2);
    P(a) = mean(abs(Y).^2);
end