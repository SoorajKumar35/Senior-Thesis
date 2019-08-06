clc;
clear;

% Import sources and recorndings of ultrasonic linear chirps

input_file = 'audible_chirp_100ms_96khz.wav';
recording_file = '1_rec_audible_chirp_100ms_96khz.wav';
[sources, Fs_rec] = audioread(input_file);
[recordings, Fs] = audioread(recording_file);
recordings = recordings(1:size(sources,1), :);

[num_samples, num_channels] = size(recordings);
num_sources = 1;
sig_duration = 100e-3;
zero_duration =  2.5;
dft_max_len = 8192;
N = Fs_rec*sig_duration;

% Get data matrix and impulse responses

offset = (zero_duration*Fs_rec);
x = nan((Fs_rec*sig_duration), num_channels, num_sources);

for i=1:num_sources
    x(:, :, i) = recordings(offset:offset+(Fs_rec*sig_duration)-1, :);
    offset = offset + (Fs_rec*(zero_duration + sig_duration));
end


%% Finding impulse responses via block-levinson algorithm
h = zeros(n, num_channels, num_sources);
offset = (zero_duration*Fs_rec);
for src_idx = 1:num_sources
   x_src = x(:, :, src_idx);
   bt_mat = create_block_toeplitz_mat(sources(offset:offset+n-1, src_idx));
   h_at_src = block_levinson(x_src, bt_mat);
   h(:, :, src_idx) = h;
   offset = offset + (Fs_rec*(zero_duration + sig_duration));
end

%%
% Calculate the impulse response via least-squares solution
% rec_samples = Fs_rec*11;
% x_conv_mat = zeros(rec_samples, 2*rec_samples);
% for i=1:rec_samples
%     x_conv_mat(i, i:i+rec_samples) = x(:, 1, 1);
% end

%% Downsampling sources before used in FFT method

% % Filter signal to remove possibility of aliasing
% lopf = fir1(48, 0.50);
% s_fil = filter(lopf, 1, s);
% 
% % Downsample signal
% s_fil = s_fil(1:2:end, :);

a = nan(N/2, num_channels , num_sources);
offset = (zero_duration*Fs_rec);

for n = 1:num_sources
    
%     S = fft(sources(offset:offset+(Fs_rec*sig_duration)-1,7));
    s = sources(offset:offset+(Fs_rec*sig_duration)-1,n);
    offset = offset + (Fs_rec*(zero_duration + sig_duration));
    
%     % Filter signal to remove possibility of aliasing and introducing
%     % artificats
%     low_pass_filter = fir1(48, 0.50);
%     s_fil = filter(low_pass_filter, 1, s);
%     x_fil = filter(low_pass_filter, 1, x(:, :, n));
%     
%     % Downsample by half
%     s_fil = s_fil(1:2:end, :);
%     S_fil = fft(s_fil, dft_max_len);
    
   % Downsample the source signal and the recording
   s_fil = decimate(s, 2, 'fir');
   x_fil = nan(floor(N/2), 5);
   for chan = 1:num_channels
        x_fil(:, chan) = decimate(x(:, chan, n), 2, 'fir');
   end

%      s_fil = s;
%      x_fil = x(:, :, n);

%     t = (0:length(s)-1)/Fs;
%     ys = ylim;
%     plot(spectrogram(outlo));
%     plot(t,outlo)
%     title('Lowpass Filtered Signal');
%     xlabel('Time (s)');
%     ylim(ys);

    S_fil = fft(s_fil);
    X = fft(x_fil);
    A = bsxfun(@rdivide,X,S_fil);
    a(:,:,n) = ifft(A,'symmetric');
    a(:, :, n) = real(a(:, :, n));
    
%     subplot(4,1,1);
    plot(a(:, :, n));
    title('Impulse response');
    xlabel('n');
    ylabel('a[n]');
    
%     subplot(4,1,2);
%     plot(atan(a(:, :, n)));
%     title("Filtered Impulse response");
%     xlabel("n");
%     ylabel("Phase - Angles (radians)");
%     
%     subplot(4,1,3);
%     plot(mag2db(abs(a(:, :, n))));
%     title("Filtered Impulse response");
%     xlabel("n");
%     ylabel("Magnitude (db)");

    window_size = 100;
    overlap_size = 20;
    
%     subplot(4,1,3);
%     spectrogram(x(:, 1, n), window_size, overlap_size, dft_max_len, 'yaxis');
%     title("Spectogram of recorded signal");
%     xlabel("t");
%     ylabel("f");
%     
%     subplot(4,1,4);
%     spectrogram(s_fil, window_size, overlap_size, dft_max_len, 'yaxis');
%     title("Spectogram of filtered source signal");
%     xlabel("t");
%     ylabel("f");
    
end

%% 

source_noise_signal = audioread('1rec_DemoGNoise_Source.wav');
% source_noise_signal = [source_noise_signal, source_noise_signal(:, 2)];
% Use impulse responses to get beamformer weights
w = easy_beamformer(a, source_noise_signal);

noisy_desired_signal_file = '1rec_DemoSourceWNoise_Source.wav';
[noisy_desired_signal, Fs_noisy_desired_signal] = audioread(noisy_desired_signal_file);


% % Apply beamformer to recordings to get better signal
y = apply_array_filter(w, noisy_desired_signal);
% disp(y);

[x, ~] = extract_multichannel(sources, recordings);
