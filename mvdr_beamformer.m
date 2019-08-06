sources_filenames = ["Ul chirp w spaces/ultrasonic_chirp_w_spaces.wav"];
recordings_filenames = ["Ul chirp w spaces/1_rec_ultrasonic_chirp_w_spaces.wav"];

[recordings, Fs_rec] = audioread(char(recordings_filenames(1)));
recordings = recordings(:, flip(1:8));
[sources, Fs] = audioread(char(sources_filenames(1)));

[x, a] = extract_multichannel(sources, recordings);

%% Now that we have the impulse responses, we can calculate the correlation/covariance matrix

% First way - for each source, we find the corresponding covariance mat

R_inter = zeros(size(a, 1), size(a, 1), size(a, 3));
for src=1:8
    R_inter = zeros(size(a, 1), size(a, 1), size(a, 2));
    for mic=1:8
        R_inter(:, :, mic) = (rms(sources(:, src))^2)*a(:, mic, src)*a(:, mic, src)';
    end
    R = sum(R_inter, 3);
end


% Second way - we find the mean R matrix over time via the use of the STFT








