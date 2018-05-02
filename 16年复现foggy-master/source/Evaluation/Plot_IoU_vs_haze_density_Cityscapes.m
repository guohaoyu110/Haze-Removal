%% Plot performance measured with the IoU metric versus haze density.

close all;
clear;

addpath('../Haze_simulation');
performance_logs_directory =...
    '../../../data/SYNTHIA_RAND_CITYSCAPES/Evaluation/DCN_Cityscapes/';
performance_logs_names =...
    file_full_names_in_directory(performance_logs_directory);
number_of_configurations = length(performance_logs_names);

% Initialize plotted vectors.
beta = zeros(1, number_of_configurations);
mean_IoU_class = zeros(1, number_of_configurations);
mean_IoU_category = zeros(1, number_of_configurations);

% Loop over performance logs and collect evaluation metrics.
for i = 1:number_of_configurations
    % Get value of beta for current configuration based on the log file name.
    performance_log_name = performance_logs_names{i};
    beta(i) = beta_from_Cityscapes_performance_log_name(performance_log_name);
    
    % Get mean IoU for Cityscapes categories and classes from log file.
    [mean_IoU_category(i), mean_IoU_class(i)] =...
    mean_IoU_scores_from_Cityscapes_performance_log_file(performance_log_name);    
end

% Sort the collected values of beta and apply the same permutation to the
% performance metrics values, for proper plotting subsequently.
[beta, perm] = sort(beta);
mean_IoU_class = mean_IoU_class(perm);
mean_IoU_category = mean_IoU_category(perm);

% Plot.
figure;
plot(beta, mean_IoU_class, '-o', 'LineWidth', 2, 'MarkerSize', 5,...
    'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
ylim([0, max(mean_IoU_class)]);
axes = gca;
y_tick_values = axes.YTick;
y_tick_step = y_tick_values(end) - y_tick_values(end - 1);
ylim([0, y_tick_values(end) + y_tick_step]);
xlabel('\beta');
title('Mean IoU for classes');

figure;
plot(beta, mean_IoU_category, '-o', 'LineWidth', 2, 'MarkerSize', 5,...
    'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
ylim([0, max(mean_IoU_category)]);
axes = gca;
y_tick_values = axes.YTick;
y_tick_step = y_tick_values(end) - y_tick_values(end - 1);
ylim([0, y_tick_values(end) + y_tick_step]);
xlabel('\beta');
title('Mean IoU for categories');

