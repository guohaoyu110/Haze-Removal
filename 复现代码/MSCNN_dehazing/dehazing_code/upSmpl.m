function Y = upSmpl(X, proc)

if strcmp(proc, 'fwd') % upsampling
    
    Y = zeros(size(X,1)*2, size(X,2)*2, size(X,3), size(X,4), 'like', X);
    Y(1:2:end, 1:2:end, :, :) = X;
    Y(1:2:end, 2:2:end, :, :) = X;
    Y(2:2:end, 1:2:end, :, :) = X;
    Y(2:2:end, 2:2:end, :, :) = X;
    
else

    Y = vl_nnpool(X, [2 2], 'pad', 0, 'stride', 2, 'method', 'avg') * 4;
    
end
