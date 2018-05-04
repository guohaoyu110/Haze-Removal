function beta_value =...
    beta_from_Cityscapes_performance_log_name(performance_log_name)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[~, log_name_plain] = fileparts(performance_log_name);
log_name_parts = strsplit(log_name_plain, '-');
if strcmp(log_name_parts{1}, 'clean')
    beta_value = 0;
else
    beta_name_value = strsplit(log_name_parts{1}, '_');
    beta_value = str2double(beta_name_value{2});
end

end

