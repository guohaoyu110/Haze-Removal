
function rImg = estimate_J(HazeImg, t, A, delta)
% dehaze an image given t and A

%
t = max(abs(t), 0.0001).^delta;

% extropolation to dehaze
HazeImg = double(HazeImg);
if length(A) == 1
    A = A * ones(3, 1);
end
R = (HazeImg(:, :, 1) - A(1)) ./ t + A(1);  %R = max(R, 0); R = min(R, 255);
G = (HazeImg(:, :, 2) - A(2)) ./ t + A(2);  %G = max(G, 0); G = min(G, 255);
B = (HazeImg(:, :, 3) - A(3)) ./ t + A(3);   %B = max(B, 0); B = min(B, 255);
rImg = cat(3, R, G, B);%S ./ 255;