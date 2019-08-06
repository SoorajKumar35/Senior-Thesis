function bl_mat = create_block_toeplitz_mat(src_at_idx)
    [N, ~] = size(src_at_idx);
    bl_mat = zeros(N, N);    
    for col=1:N
       bl_mat(col, col:end) = src_at_idx(col:end); 
    end
end
