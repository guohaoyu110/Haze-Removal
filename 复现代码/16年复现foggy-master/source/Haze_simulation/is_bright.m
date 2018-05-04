function bright = is_bright(L, theta)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

bright = rgb2gray(L) >= theta;

end

