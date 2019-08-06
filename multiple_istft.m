function x = multiple_istft(X,frame_length,frame_shift)
% MULTIPLE_ISTFT Computes the multidimensional inverse short-time Fourier 
% transform (ISTFT) of X along the first two dimensions of X.
%
% x = MULTIPLE_ISTFT(X) inverts the STFT assuming 50% overlap and no zero
% padding. The input X has dimension (num bins x num frames x D1 x D2 x ...
% x D_last) and the output x has dimension (num samples x D1 x D2 x ... x
% D_last). If the length of the input to MULTIPLE_STFT was a multiple of
% the frame shift, then MULTIPLE_ISTFT(MULTIPLE_STFT(x)) = x. Otherwise,
% the output will have extra zero samples at the end.
%
% x = MULTIPLE_ISTFT(X,FRAME_LENGTH,FRAME_SHIFT) inverts the STFT assuming 
% the specified frame length and shift.


d = size(X);
num_bins = d(1);
num_frames = d(2);
num_channels = prod(d(3:end));
dft_length = 2*(num_bins-1);

% Default parameters
if nargin < 3, frame_length = dft_length; end
if nargin < 2, frame_shift = frame_length/2; end


% Vectorize dimensions of multidimensional STFT matrix
X = reshape(X,[num_bins,num_frames,num_channels]);


% Compute output signal size
xlen = dft_length + (num_frames-1)*frame_shift;
x = zeros(xlen,num_channels);

% Compute ISTFT
X = permute(X,[1 3 2]); % num_bins x num_channels x num_frames
for t = 1:num_frames
    start_ind = 1+(t-1)*frame_shift;
    end_ind = dft_length+(t-1)*frame_shift;
    
    % Conjugate symmetric IDFT
    X_block = [X(:,:,t); conj(X(end-1:-1:2,:,t))];
    x_block = ifft(X_block,'symmetric');
    
    % Overlap-add
    x(start_ind:end_ind,:) = x(start_ind:end_ind,:) + x_block;
end

% Remove padding added to forward transform
x = x(dft_length+1:end-dft_length,:);

% Reassemble dimensions
x = reshape(x,[size(x,1) d(3:end)]);

% Undo normalization from multiple_stft
w = hanning(frame_length,'periodic');
x = x * rms(w)/mean(w)/sqrt(frame_length)*frame_shift;