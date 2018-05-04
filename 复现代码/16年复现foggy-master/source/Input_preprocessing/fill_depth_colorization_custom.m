% Preprocesses the kinect depth image using a gray scale version of the
% RGB image as a weighting for the smoothing. This code is a slight
% adaptation of Anat Levin's colorization code:
%
% See: www.cs.huji.ac.il/~yweiss/Colorization/
%
% Args:
%   imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
%            be between 0 and 1.
%   imgDepth - HxW matrix, the depth image for the current frame in
%              absolute (meters) space.
%   alpha - a penalty value between 0 and 1 for the current depth values.
function denoisedDepthImg = fill_depth_colorization_custom(imgRgb, imgDepth, alpha)
error(nargchk(2, 3, nargin));
if nargin < 3
    alpha = 1;
end

imgIsNoise = (imgDepth == 0 | imgDepth == 10);

maxImgAbsDepth = max(imgDepth(~imgIsNoise));
imgDepth = imgDepth ./ maxImgAbsDepth;
imgDepth(imgDepth > 1) = 1;

assert(ndims(imgDepth) == 2);
[H, W] = size(imgDepth);
numPix = H * W;

indsM = reshape(1:numPix, H, W);

knownValMask = ~imgIsNoise;

grayImg = rgb2gray(imgRgb);

winRad = 1;

len = 0;
absImgNdx = 0;
cols = zeros(numPix * (2*winRad+1)^2,1);
rows = zeros(numPix * (2*winRad+1)^2,1);
vals = zeros(numPix * (2*winRad+1)^2,1);
gvals = zeros(1, (2*winRad+1)^2);

for j = 1 : W
    for i = 1 : H
        absImgNdx = absImgNdx + 1;
        
        % Counts the number of points in the current window.
        nWin = 0;
        for ii = max(1, i-winRad) : min(i+winRad, H)
            for jj = max(1, j-winRad) : min(j+winRad, W)
                if ii == i && jj == j
                    continue;
                end
                
                len = len+1;
                nWin = nWin+1;
                rows(len) = absImgNdx;
                cols(len) = indsM(ii,jj);
                gvals(nWin) = grayImg(ii, jj);
            end
        end
        
        curVal = grayImg(i, j);
        gvals(nWin+1) = curVal;
        c_var = mean((gvals(1:nWin+1)-mean(gvals(1:nWin+1))).^2);
        
        csig = c_var*0.6;
        mgv = min((gvals(1:nWin)-curVal).^2);
        if csig < (-mgv/log(0.01))
            csig=-mgv/log(0.01);
        end
        
        if csig < 0.000002
            csig = 0.000002;
        end
        
        gvals(1:nWin) = exp(-(gvals(1:nWin)-curVal).^2/csig);
        gvals(1:nWin) = gvals(1:nWin) / sum(gvals(1:nWin));
        vals(len-nWin+1 : len) = -gvals(1:nWin);
        
        % Now the self-reference (along the diagonal).
        len = len + 1;
        rows(len) = absImgNdx;
        cols(len) = absImgNdx;
        vals(len) = 1; %sum(gvals(1:nWin));
    end
end

% Removes trailing zeros from the end of the vectors. These zeros are due to the
% fact that pixels lying on the image boundary have fewer neighbors than
% (2 * winRad + 1) ^ 2.
vals = vals(1:len);
cols = cols(1:len);
rows = rows(1:len);

% Create sparse matrix |A| which contains the weights that appear in the
% quadratic objective, by putting its non-zero elements held in vector |vals| at
% the position specified by the corresponding elements of vectors |rows| and
% |cols|.
A = sparse(rows, cols, vals, numPix, numPix);

% rows = 1:numPix;
% cols = 1:numPix;
% vals = knownValMask(:) * alpha;
% G = sparse(rows, cols, vals, numPix, numPix);
% new_vals = (A + G) \ (vals .* imgDepth(:));

% Create sparse matrix |M| of linear constraints.
M = speye(numPix);
M = M(knownValMask(:), :);

% Form vector |d| of known depth values which appears on the right-hand side of
% the linear constraints.
d = imgDepth(knownValMask(:));

% Solve sparse linear system using MATLAB backslash operator to compute the
% unknown depth values and the Lagrange multipliers.
K = nnz(knownValMask);
denoisedDepth = [A.' * A, M.'; M, sparse(K, K)] \ [zeros(numPix, 1); d];

% Bring result back to proper shape and scale.
denoisedDepthImg = reshape(denoisedDepth(1:numPix), [H, W]);
denoisedDepthImg = denoisedDepthImg * maxImgAbsDepth;

end
