%% P-wave Parameter Analysis Script - Binary Comparison with Legends Only
clear; clc; close all;

%% Parameters
function p = get_base_parameters()
    % VDP para
    a1 = 40; a2 = 50; a3 = 50;
    u11 = 0.83; u21 = 0.83; u31 = 0.83;
    u12 = -0.83; u22 = -0.83; u32 = -0.83;
    f1 = 22; f2 = 8.4; f3 = 1.5;
    d1 = 3; d2 = 3; d3 = 3;
    e1 = 3.5; e2 = 5; e3 = 12;
    K_SA_AV = f1; 
    K_AV_HP = f1;
    tau = 0.092;
    
    % FHN Para
    kk1 = 2000.0; kk2 = 400.0; kk3 = 1300.0; kk4 = 2000.0;
    c1 = 0.26; c2 = 0.26; c3 = 0.12; c4 = 0.10;
    d1_ode = 0.4; d2_ode = 0.4; d3_ode = 0.09; d4_ode = 0.1;
    h1 = 0.004; h2 = 0.004; h3 = 0.008; h4 = 0.008;
    g1 = 1.0; g2 = 1.0; g3 = 1.0; g4 = 1.0;
    w11 = 0.13; w12 = 1.0;
    w21 = 0.19; w22 = 1.0;
    w31 = 0.12; w32 = 1.1;
    w41 = 0.22; w42 = 0.8;
    
    % Additional Para
    a_K = 0; a_Ca = 0;
    K_0 = 4.0; K_1 = 4.5;
    Ca_0 = 1.0; Ca_1 = 1.2;
    b1 = 0; b2 = 0; b3 = 0.015; b4 = 0;
    
    p = [a1, a2, a3, u11, u12, u21, u22, u31, u32, f1, f2, f3, d1, d2, d3, e1, e2, e3, K_SA_AV, K_AV_HP, tau, ...
         kk1, kk2, kk3, kk4, c1, c2, c3, c4, d1_ode, d2_ode, d3_ode, d4_ode, h1, h2, h3, h4, g1, g2, g3, g4, ...
         w11, w12, w21, w22, w31, w32, w41, w42, ...
         a_K, a_Ca, K_0, K_1, Ca_0, Ca_1, b1, b2, b3, b4, m1, m2];
end

%% Simulation Function
function [t, ECG, P_wave, waves] = run_simulation(p)
    tau = p(21);
    u0_combined = [0.1, 0.1, 0.2, 0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01];
    
    tspan = [0.0, 10.0];
    sol = dde23(@(t,y,Z) pacemaker_fhn(t,y,Z,p), tau, u0_combined, tspan);
    t = linspace(0, 10, 2000);
    sol_combined = deval(sol, t);
    
    % Extract waves
    SA = sol_combined(1, :);
    AV = sol_combined(3, :);
    PKJ = sol_combined(5, :);
    P_wave = sol_combined(7, :);    
    Ta_wave = sol_combined(9, :);
    QRS_wave = sol_combined(11, :);
    T_wave = sol_combined(13, :);
    
    ECG = P_wave - Ta_wave + QRS_wave + T_wave;
    ATR = P_wave - Ta_wave;
    VTR = QRS_wave + T_wave;
    
    waves = struct('SA', SA, 'AV', AV, 'PKJ', PKJ, 'P_wave', P_wave, ...
                   'Ta_wave', Ta_wave, 'QRS_wave', QRS_wave, 'T_wave', T_wave, ...
                   'ATR', ATR, 'VTR', VTR);
end

%% Plot
function create_binary_comparison_figure(param_info, base_p, fig_num)
    param_values = [min(param_info.values), max(param_info.values)];
    param_index = param_info.index;
    param_display_name = param_info.name;
    default_value = base_p(param_index);
    
    figure(fig_num);
    set(gcf, 'Position', [150 + (fig_num-1)*50, 150 + (fig_num-1)*50, 800, 600]);
    
    blue_color = [0, 0.4470, 0.7410];
    red_color = [0.8500, 0.3250, 0.0980];
    
    line_width = 1.0;
    
    if abs(param_values(1) - default_value) < abs(param_values(2) - default_value)
        default_idx = 1;
        modified_idx = 2;
    else
        default_idx = 2;
        modified_idx = 1;
    end
    
    legend_entries = cell(2, 1);
    
    subplot(2, 1, 1);
    hold on;
    
    title_str = sprintf('%s=%.3f; %s=%.3f', param_display_name, param_values(1), ...
                       param_display_name, param_values(2));
    
    for j = 1:length(param_values)
        p_temp = base_p;
        p_temp(param_index) = param_values(j);
        
        try
            [t, ECG, P_wave, waves] = run_simulation(p_temp);
            
            if j == default_idx
                color = blue_color;
                line_style = '-';  
                legend_entries{j} = sprintf('%s=%.3f', param_display_name, param_values(j));
            else
                color = red_color;
                line_style = '--'; 
                legend_entries{j} = sprintf('%s=%.3f', param_display_name, param_values(j));
            end
            
            plot(1:length(ECG), ECG, 'Color', color, 'LineStyle', line_style, ...
                 'LineWidth', line_width, 'DisplayName', legend_entries{j});
            
        catch ME
            fprintf('参数 %s=%.3f 仿真失败: %s\n', param_display_name, param_values(j), ME.message);
        end
    end
    
    xlabel('Time Index');
    ylabel('ECG');
    title(title_str, 'FontSize', 12);
    legend('show', 'Location', 'best', 'FontSize', 10);
    grid on;
    xlim([0, 2000]);
    ylim([-0.1, 0.3]);
    
    subplot(2, 1, 2);
    hold on;
    
    for j = 1:length(param_values)
        p_temp = base_p;
        p_temp(param_index) = param_values(j);
        
        try
            [t, ECG, P_wave, waves] = run_simulation(p_temp);
            
            if j == default_idx
                color = blue_color;
                line_style = '-';
            else
                color = red_color;
                line_style = '--';
            end
            
            plot(1:length(P_wave), P_wave, 'Color', color, 'LineStyle', line_style, ...
                 'LineWidth', line_width, 'DisplayName', legend_entries{j});
            
        catch ME
        end
    end
    
    xlabel('Time Index');
    ylabel('Pwave');
    title(title_str, 'FontSize', 12);
    legend('show', 'Location', 'best', 'FontSize', 10);
    grid on;
    xlim([0, 2000]);
    ylim([-0.05, 0.25]);
    
    saveas(gcf, sprintf('Pwave_Parameter_%s_Binary_Comparison.png', param_display_name));
    fprintf('Plots saved: Pwave_Parameter_%s_Binary_Comparison.png\n', param_display_name);
end

%% main script
base_p = get_base_parameters();

% Parameters and indices
p_wave_params = struct();
p_wave_params.kk1 = struct('index', 22, 'values', [1000, 2000, 3000], 'name', 'kk1');
p_wave_params.c1 = struct('index', 26, 'values', [0.16, 0.26, 0.36], 'name', 'c1');
p_wave_params.d1_ode = struct('index', 30, 'values', [0.3, 0.4, 0.5], 'name', 'd1ode');
p_wave_params.h1 = struct('index', 34, 'values', [0.002, 0.004, 0.1], 'name', 'h1');
p_wave_params.g1 = struct('index', 38, 'values', [0.1, 1.0, 1.5], 'name', 'g1');
p_wave_params.w11 = struct('index', 42, 'values', [0.1, 0.13, 0.3], 'name', 'w11');
p_wave_params.w12 = struct('index', 43, 'values', [0.5, 1.0, 1.5], 'name', 'w12');
p_wave_params.b1 = struct('index', 56, 'values', [0, 0.0, 0.05], 'name', 'b1');
p_wave_params.m1 = struct('index', 60, 'values', [-10, 0.0, 10], 'name', 'm1');
p_wave_params.m2 = struct('index', 61, 'values', [-10, 0.0, 10], 'name', 'm2');

param_names = fieldnames(p_wave_params);
num_params = length(param_names);


for i = 1:num_params
    param_name = param_names{i};
    param_info = p_wave_params.(param_name);
    
    fprintf('Generating para %s Comparison plot\n', param_info.name);
    
    create_binary_comparison_figure(param_info, base_p, i);
end

%% Summary
fprintf('\n=== Analysis Completed ===\n');
fprintf('=========================================================\n');
fprintf('Parameter\t\tIndex\t\tDefault\t\tMin\t\tMax\n');
fprintf('=========================================================\n');
for i = 1:num_params
    param_name = param_names{i};
    param_info = p_wave_params.(param_name);
    default_val = base_p(param_info.index);
    min_val = min(param_info.values);
    max_val = max(param_info.values);
    fprintf('%s\t\t\t%d\t\t%.3f\t\t%.3f\t\t%.3f\n', ...
            param_info.name, param_info.index, default_val, min_val, max_val);
end
fprintf('=========================================================\n');