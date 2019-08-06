function [X,f,t] = multiple_stft(x,frame_length,frame_shift,dft_length,fs)
% MSTFT Computes the multidimensional short-time Fourier transform (STFT) over the first dimension of x.
%
% X = MULTIPLE_STFT(x) computes the STFT of x using a frame length of 1024
% samples, frame shift of 512 samples (50% overlap), and DFT length of 1024
% samples (no zero padding). The time-domain signal x has dimension (signal
% length x D1 x D2 x ... x D_last) and the output has dimension (num bins x
% num frames x D1 x D2 x ... x D_last). The input x must be real-valued,
% since the output includes only the positive frequencies (num bins = DFT
% length/2 + 1).
%
% X = MULTIPLE_STFT(x,FRAME_LENGTH) computes the STFT using frames of length
% FRAME_LENGTH samples, frame shift of FRAME_LENGTH/2 (50% overlap), and DFT
% length of FRAME_LENGTH (no zero padding).
% 
% X = MULTIPLE_STFT(x,FRAME_LENGTH,FRAME_SHIFT) uses a frame shift of length
% FRAME_SHIFT, which should almost always be either 1/2 or 1/4 of FRAME_LENGTH.
%
% X = MULTIPLE_STFT(x,FRAME_LENGTH,FRAME_SHIFT,DFT_LENGTH) uses a DFT
% length of DFT_LENGTH samples, which must be greater than or equal to
% FRAME_LENGTH. If it is greater, zeros are padded to each frame.
%
% [X,F,T] = MULTIPLE_STFT(x,FRAME_LENGTH,FRAME_SHIFT,DFT_LENGTH,FS)
% computes the frequencies F, in Hertz, corresponding to the rows of X and
% the times T, in seconds, corresponding to the columns of X, using the
% sample rate FS.
%
% [X,F,T] = MULTIPLE_STFT(x,____) computes the frequencies and times based
% on a sample rate of 16000 samples per second.
%
% MULTIPLE_STFT scales the outputs so that if the input were white noise,
% then the mean squared values of each sample of x and of each sample of X
% are the same. This is not standard for the STFT, but is convenient for
% visualizing the transformed data and designing filters.

% Default arguments
if nargin < 2, frame_length = 1024; end
if nargin < 3, frame_shift = frame_length/2; end
if nargin < 4, dft_length = frame_length; end
if nargin < 5, fs = 16000; end

% Collapse higher dimensions of the matrix into num_samples x N
d = size(x);
num_samples = d(1);
num_channels = prod(d(2:end));
x = reshape(x,[num_samples num_channels]);

% Zero pad to prevent errors at the edges
x = [zeros(dft_length,num_channels);x;zeros(dft_length,num_channels)];


% Compute the zero-padded von Hann window
% Padding on both sides isn't strictly necessary, but it makes the time
% values line up nicely.
win = hanning(frame_length,'periodic');
pad_length = (dft_length-frame_length)/2;
win = [zeros(pad_length,1); win; zeros(pad_length,1)];

% Scale factor to maintain same energy per sample at output
% This scaling is not standard, but I find it convenient
scale_factor = 1/sqrt(sum(win.^2));
x = x*scale_factor;

% Allocate output matrix
num_frames = 1 + floor((length(x)-dft_length)/frame_shift);
num_bins = 1 + dft_length/2;
X = zeros(num_bins,num_channels,num_frames);

for t = 1:num_frames
    start_ind = 1+frame_shift*(t-1);
    end_ind = dft_length+frame_shift*(t-1);
    
    % Apply window
    xw = bsxfun(@times,win,x(start_ind:end_ind,:));
    
    % Compute the DFT
    XW = fft(xw);
    
    % Since x is real-valued, we can throw away negative frequencies
    X(:,:,t) = XW(1:num_bins,:); 
end

% Restore to original matrix dimensions
X = permute(X,[1 3 2]); % F x T x N
X = reshape(X,[size(X,1) size(X,2) d(2:end)]); % F x T x ...

if nargout > 1
    % calculate the time and frequency vectors
    t = (dft_length/2:frame_shift:dft_length/2+(num_frames-1)*frame_shift)/fs;
    f = (0:num_bins-1)*fs/dft_length;
end