function [x,a] = extract_multichannel(sources,recordings)
% EXTRACT_MULTICHANNEL extracts multichannel data from a recording of
% separate source signals
%
% X = EXTRACT_MULTICHANNEL(SOURCES,RECORDINGS) extracts a (duration x num 
% channels x num sources) multidimensional data matrix X from a (recording 
% len x num sources) vector SOURCES in which each source is activated sequentially
% in a separate channel and the resulting (recording len x num microphones) 
% array recording RECORDINGS. The source start and end points are detected 
% automatically from the source signal. The source and recording data can 
% have different durations, but are assumed to start simultaneously. Each
% source must be nonzero only over one interval (i.e., not repeated) and no
% two sources may be active at the same time.
%
% [X,A] = EXTRACT_MULTICHANNEL(SOURCES,RECORDINGS) will also estimate the
% impulse responses from the sources to the array. For this to work
% reliably, the source signals should be appropriate test signals such as
% chirps or pseudorandom noise.
%
% Note: The inputs can also be provided as filenames for multichannel audio
% files.
%
% The data in SOURCES should look like:
%       Src 1: -AAA---------------------
%       Src 2: -----BBB-----------------
%       ...
%       Src N: ---------------------CCC-
%
% The data in RECORDINGS will look like:
%       Mic 1: -AAA-BBB-------------CCC-
%       Mic 2: -AAA-BBB-------------CCC-
%       ...
%       Mic M: -AAA-BBB-------------CCC-
%
% The output data in X will look like:
%         |--  Mic 1: AAA-
%  Src 1: |    Mic 2: AAA-
%         |    ...
%         |--  Mic 3: AAA-
%
%         |--  Mic 1: BBB-
%  Src 2: |    Mic 2: BBB-
%         |    ...
%         |--  Mic 3: BBB-
%  ...
%         |--  Mic 1: CCC-
%  Src N: |    Mic 2: CCC-
%         |    ...
%         |--  Mic 3: CCC-

% Read audio files if the inputs are filenames
if ischar(sources), sources = audioread(sources); end
if ischar(recordings), recordings = audioread(recordings); end

num_sources = size(sources,2);
num_channels = size(recordings,2);

% Find start times of the test signals in each channel
start_times = nan(num_sources,1);
for n = 1:num_sources
    start_times(n) = find(sources(:,n),1);
end

% Find duration of test signal
test_duration = min(diff(sort(start_times)));

% Extract signals from each channel
x = nan(test_duration,num_channels,num_sources);

if (start_times(size(start_times,1))+test_duration)>size(recordings,1)
    recordings = [recordings; zeros(start_times(size(start_times,1))+test_duration-size(recordings,1),8)];
end
for n = 1:num_sources
    x(:,:,n) = recordings(start_times(n):start_times(n)+test_duration-1,:);
end

% Estimate impulse responses
if nargout > 1
    a = nan(test_duration,num_channels,num_sources);
    sources = [sources;zeros(test_duration,num_sources)]; % Add buffer to account for spacing between references
    for n = 1:num_sources
        S = fft(sources(start_times(n):start_times(n)+test_duration-1,n));
        X = fft(x(:,:,n));
        A = bsxfun(@rdivide,X,S);
        a(:,:,n) = ifft(A,'symmetric');
    end
end 