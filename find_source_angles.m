clear;
clc;
[recordings, Fs_rec] = audioread('5_rec_7th_src_chrip.wav');
[sources, Fs] = audioread('7th_src_chrip.wav');

% [x, ~] = extract_multichannel(sources, recordings);
[num_samples, num_channels] = size(recordings);
num_sources = 5;
sig_duration = 11;
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


angles = 0:1:179;
angles = angles * (pi/180);
% source_one_channel_allmics = sources(:,:,1);
% S = fft(source_one_channel_allmics);

% a_src_1 = a(:, :, 1);
% A = fft(a_src_1);
% db_scale = mag2db(abs(A/size(A, 1));
% T = size(a_src_1,1)/Fs;
% freq = (1:size(a_src_1,1))/T;
% 
% colors = ['y', 'm', 'c', 'r', 'g', 'b', 'w', 'k'];
% 
% for i=1:4
%     subplot(2,2,i)
% %     plot(freq, mag2db((A(1:size(A,1),i))), 'b')
%     plot(1:size(a_src_1,1), mag2db(a_src_1(:,i)))
%     title("Impulse response");
%     xlabel("Time index");
%     ylabel("Value at index");
% end
for i=1:8
    disp(i)
    P = sooraj_micarray_srp_phat(x(:,:,i), angles, Fs_rec);
%     subplot(4, 2, i)
    plot(1:size(angles, 2), P)
    title("Power vs angle");
    xlabel("Angle");
    ylabel("Power");
    [~, j] = max(P);
    disp(angles(j)*(180/pi))
end
