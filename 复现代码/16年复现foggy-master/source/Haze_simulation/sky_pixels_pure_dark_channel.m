function number_of_pure_sky_pixels_dark_channel =...
    sky_pixels_pure_dark_channel(ground_truth_labels, sky_label, se)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

is_sky = ground_truth_labels == sky_label;
is_sky_eroded = imerode(is_sky, se);
number_of_pure_sky_pixels_dark_channel = nnz(is_sky_eroded);

end

