function y = apply_array_filter(filters,signals)
% APPLY_ARRAY_FILTER Apply a filter-and-sum beamformer to a multichannel
% signal.
%
% Y = APPLY_ARRAY_FILTER(W,X) applies the multichannel filter coefficients
% in W to the multichannel signals in X. W has dimension (filter length x
% number of channels x number of outputs) and X has dimension (sequence
% length x number of channels). The output Y has dimension (sequence
% length + filter length - 1 x number of outputs).
%
% This function uses time-domain convolution to compute the filter outputs,
% so it is suitable for low-delay causal filtering. The length of the
% output is truncated to match the length of the input.
%
% If X has dimension (sequence length x number of channels x N), then
% W will be applied separately to each frame, giving an output Y with
% dimension (sequence length + filter length - 1 x number of outputs x N).
% This is useful for simulating the effect of the filter on different
% sources independently.

[filter_len,num_channels,num_outputs] = size(filters);
d = size(signals); if length(d) < 3, d(3) = 1; end
signal_length = d(1);
num_inputs = prod(d(3:end));
if size(signals,2) ~= num_channels, error('The filter expects %d channels but the input has %d channels.',num_channels,size(signals,2)); end

y = zeros(signal_length+filter_len-1,num_outputs,num_inputs);
for k = 1:num_inputs
    for n = 1:num_outputs
        for m = 1:num_channels
            ym = conv(filters(:,m,n),signals(:,m,k));
            y(:,n,k) = y(:,n,k) + ym;
        end
    end
end

if length(d) > 3, y = reshape(y,[size(signals,1),num_outputs,d(3:end)]);

end