function w = easy_beamformer(target_ir,noise)
% EASY_BEAMFORMER Generate a simple beamformer from impulse response data.
%
% W = EASY_BEAMFORMER(TARGET_IR) creates one or more matched filters
% pointed at the source(s) with impulse responses in TARGET IR. TARGET_IR
% has dimension (response length x number of channels x number of outputs).
% The array filter W has dimension (filter length x number of channels x
% number of outputs).
%
% W = EASY_BEAMFORMER(TARGET_IR,NOISE) designs filters to suppress a
% particular distribution of noise sources. NOISE can be either a (filter
% length x number of channels x number of outputs) matrix of impulse
% responses or a (signal length x number of channels) recording of noise.
% If a noise sample is provided, it should be many times longer than the
% filter length to ensure good results.
%
% The beamformer is a speech-distortion-weighted multichannel Wiener filter
% designed to emphasize target distortion over interference suppression and
% noise reduction. The filter is designed to reproduce the source as
% received by the first sensor.

% Tuning parameters
MAX_LEN = 8192;
SDW = 1e+1; % Weight to emphasize source distortion over interference suppression.
DIAGONAL_LOADING = 1e-2; % Helps with impulse response estimation errors and motion.

% Housekeeping
[ir_len,num_channels,num_outputs] = size(target_ir);
dft_len = min(ir_len*2,MAX_LEN);

% Find source transfer function(s)
for n = 1:num_outputs
    target_ir(:,:,n) = target_ir(:,:,n) / sqrt(mean(sum(target_ir(:,:,n).^2,1),2)); % Normalize responses
end
Ht = permute(fft(target_ir,dft_len),[2 3 1]); % Ht has dimension num_channels x num_sources x dft_len

% Find interference covariance matrix
C = zeros(num_channels,num_channels,dft_len);
if nargin > 1
%     noise = noise / sqrt(sum(noise(:).^2)/num_channels);
    if size(noise,3) > 1 || size(noise,1) <= ir_len
        % Find interference cross-power spectral density from impulse responses
        Hi = permute(fft(noise,dft_len),[2 3 1]); % Hi has dimensions num_channels x num_interferers x dft_len
        Hi = Hi./sqrt(mean(sum(mean(abs(Hi).^2,1),2),3));
    else
        % Find empirical noise cross-power spectral density from recording
        Hi = permute(multiple_stft(noise,dft_len),[3 2 1]); % Hi has dimensions num_channels x num_frames x dft_len            
        Hi = cat(3,Hi,conj(Hi(:,:,end-1:-1:2)));
        Hi = bsxfun(@rdivide,Hi,sqrt(sum(mean(abs(Hi).^2,1),2)));
    end
    % Compute cross-power spectral density and add diagonal loading and noise terms
    for f = 1:dft_len
        C(:,:,f) = Hi(:,:,f)*Hi(:,:,f)';
        C(:,:,f) = C(:,:,f) + DIAGONAL_LOADING*eye(num_channels);
    end
else
    C = eye(num_channels);
end

% Compute beamformer
W = zeros(num_outputs,num_channels,dft_len);
for f = 1:dft_len
    for n = 1:num_outputs
        W(n,:,f) = Ht(1,n,f)*Ht(:,n,f)'/(Ht(:,n,f)*Ht(:,n,f)'+C(:,:,f)/SDW);
    end
end

W = permute(W,[3 2 1]); % dft_len x num_channels x num_outputs
w = ifft(W,'symmetric');
w = circshift(w,dft_len/4);  % Add delay to make the filter causal shift so we can use non-circular convolution
w = w(1:dft_len/2,:,:);