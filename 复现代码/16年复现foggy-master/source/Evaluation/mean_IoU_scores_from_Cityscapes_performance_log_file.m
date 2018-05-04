function [mean_IoU_category, mean_IoU_class] =...
    mean_IoU_scores_from_Cityscapes_performance_log_file(performance_log_name)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

log_id = fopen(performance_log_name, 'r');
log_content = textscan(log_id, '%s', 3, 'delimiter', '\n');
log_first_three_lines = log_content{1};
line_2_parts = strsplit(log_first_three_lines{2}, {' ', ','});
mean_IoU_category = str2double(line_2_parts{end - 1});
line_3_parts = strsplit(log_first_three_lines{3}, {' ', ','});
mean_IoU_class = str2double(line_3_parts{end - 1});
fclose(log_id);

end

