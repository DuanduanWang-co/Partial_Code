% fixed: w11，w12;
% fit: kk1、b1、c1, GA, MSE
% 2025/8/28, run on server
clc; clear; close all;

%% === 参数定义 ===
base_p = [
40,50,50,0.83,-0.83,0.83,-0.83,0.83,-0.83,...
22,8.4,1.5,3,3,3,3.5,5,12,...
22,22,0.092,...
1300,400.0,1300.0,2000.0,...
0.36,0.26,0.12,0.10,...
0.4,0.4,0.09,0.1,...
0.004,0.004,0.008,0.008,...
1.0,1.0,1.0,1.0,...
0.13,1.0,0.19,1.0,0.12,1.1,0.22,0.8,...
0.15,0.0,4.0,4.5,1.0,1.0, 0.03];
u0_combined = [0.1,0.1,0.2,0.1,0.5,0.1,0.1,0.1,0.1,0.1,0.01,0.01,0.01,0.01];

%% Fixed Parameters 
w11_val = 0.126405;     
w12_val = 0.675950;      
d1_ode_val = 5.75;  % 固定 

%% Read Real Data
filename = '/home/ddwang/8_15_2025/Pwave_Output/10000032_44458630_all_pwaves.csv';
T = readtable(filename);
wave_ids = unique(T.wave_id);
used_wave_id = wave_ids(1);
T_wave = T(T.wave_id == used_wave_id, :);
t_real_ms = T_wave.time_ms;
v_real = T_wave.voltage;
[~, sort_idx] = sort(t_real_ms);
t_real_ms = t_real_ms(sort_idx);
v_real = v_real(sort_idx);

t_real = (t_real_ms - min(t_real_ms)) / 1000;
v_real = v_real - min(v_real);

% Decrease fs points
max_points = 50;
if length(t_real) > max_points
    idx_ds = round(linspace(1, length(t_real), max_points));
    t_real = t_real(idx_ds);
    v_real = v_real(idx_ds);
end

figure;
plot(t_real, v_real, 'bo-', 'DisplayName', 'Real P wave Data');
xlabel('Time(s)');
ylabel('Voltage');
title('Real P Wave Data');
grid on;

%% GA Settings
% Para: [kk1, b1, c1]
lb = [1200.0, -0.1, 0.3];    
ub = [1500.0, 1, 1.0];   
nvars = length(lb);

% GA Options
ga_options = optimoptions('ga', ...
'PopulationSize', 100, ...
'MaxGenerations', 200, ... 
'CrossoverFraction', 0.8, ... 
'MutationFcn', {@mutationadaptfeasible, 0.02}, ... 
'SelectionFcn', @selectiontournament, ... 
'EliteCount', 5, ... 
'Display', 'iter', ... 
'PlotFcn', {@gaplotbestf}, ... 
'UseParallel', false, ... 
'TolFun', 1e-8, ... 
'StallGenLimit', 50); 

% Object Function
objfun = @(x) compute_loss_enhanced( ...
    x, base_p, u0_combined, t_real, v_real, ...
    w11_val, w12_val, d1_ode_val);

%% Main Code
fprintf('GA Optimization Begins');
tic;
[x_opt_ga, fval_ga, exitflag, output] = ga(objfun, nvars, [], [], [], [], lb, ub, [], ga_options);
elapsed_time = toc;

fprintf('Results');
fprintf('Best Fit:');
fprintf('kk1 = %.6f\n', x_opt_ga(1));
fprintf('b1 = %.6f\n', x_opt_ga(2));
fprintf('c1 = %.6f\n', x_opt_ga(3));
fprintf('Min Loss = %.8f\n', fval_ga);
fprintf('Optimization Time = %.2f s\n', elapsed_time);
fprintf('Total Generations = %d\n', output.generations);

%% Local Refinement Optimization
fprintf('Local refinement optimization begins');
opts_local = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 100);
[x_opt_refined, fval_refined] = fmincon(objfun, x_opt_ga, [], [], [], [], lb, ub, [], opts_local);

fprintf('Results');
fprintf('Refined Para:\n');
fprintf('kk1 = %.6f\n', x_opt_refined(1));
fprintf('b1 = %.6f\n', x_opt_refined(2));
fprintf('c1 = %.6f\n', x_opt_refined(3));
fprintf('Loss = %.8f\n', fval_refined);

% Choose the best results
if fval_refined < fval_ga
    x_final = x_opt_refined;
    fval_final = fval_refined;
    fprintf('Use Refined Results as the Final Results');
else
    x_final = x_opt_ga;
    fval_final = fval_ga;
    fprintf('Use Original Results as the Final Results');
end

%% Comparison
p_fit = base_p;
p_fit(22) = x_final(1);  % kk1
p_fit(56) = x_final(2);  % b1
p_fit(26) = x_final(3);  % c1
p_fit(30) = d1_ode_val;  % fixed d1_ode
p_fit(42) = w11_val;     % fixed w11
p_fit(43) = w12_val;     % fixed w12

[t_pwave, p_pwave] = extract_single_pwave(p_fit, u0_combined, [0 10], 1000);

t_norm = linspace(0,1,length(t_real));
t_single_norm = linspace(0,1,length(p_pwave));
p_interp = interp1(t_single_norm, p_pwave, t_norm, 'linear', 'extrap');
p_aligned = align_peaks(p_interp, v_real);

figure;
subplot(2,1,1);
plot(t_real, v_real, 'bo-', 'DisplayName','Real P wave', 'LineWidth', 1.5);
hold on;
plot(t_real, p_aligned, 'r--', 'LineWidth', 2, 'DisplayName', 'Fitting Model');
xlabel('Time');
ylabel('Voltage');
title('Real P wave vs Fitting Model');
legend;
grid on;

% Error
p_aligned_norm = (p_aligned - min(p_aligned)) / (max(p_aligned) - min(p_aligned) + 1e-8);
p_target_norm = (v_real - min(v_real)) / (max(v_real) - min(v_real) + 1e-8);
error_signal = p_aligned_norm - p_target_norm;

subplot(2,1,2);
plot(t_real, error_signal, 'g-', 'LineWidth', 1.5);
xlabel('Time');
ylabel('Normalized Error');
title('Fitting Error');
grid on;

%% Loss Function Definition
function loss = compute_loss_enhanced(x, base_p, u0, t_target, p_target, ...
                                      w11_val, w12_val, d1_ode_val)
try
        p = base_p;
        p(22) = x(1);        % 优化 kk1
        p(56) = x(2);        % 优化 b1
        p(26) = x(3);        % 优化 c1
        p(30) = d1_ode_val;  % 固定 d1_ode
        p(42) = w11_val;     % 固定 w11
        p(43) = w12_val;     % 固定 w12
        
        [t_single, p_single] = extract_single_pwave(p, u0, [0 10], 1000);

if length(p_single) < 3 || any(isnan(p_single)) || any(isinf(p_single))
            loss = 1e6;
return;
end

        t_norm = linspace(0,1,length(t_target));
        t_single_norm = linspace(0,1,length(p_single));
        p_interp = interp1(t_single_norm, p_single, t_norm,'linear','extrap');
        p_aligned = align_peaks(p_interp, p_target);

% Normalized
        p_aligned_norm = (p_aligned - min(p_aligned)) / (max(p_aligned) - min(p_aligned) + 1e-8);
        p_target_norm = (p_target - min(p_target)) / (max(p_target) - min(p_target) + 1e-8);

% MSE
        mse_loss = mean((p_aligned_norm - p_target_norm).^2);

% Peak Error
        [~, peak_model] = max(p_aligned_norm);
        [~, peak_target] = max(p_target_norm);
        peak_error = abs(peak_model - peak_target) / length(p_target_norm);

% Corr Error
        corr_coeff = corrcoef(p_aligned_norm, p_target_norm);
        corr_loss = 1 - abs(corr_coeff(1,2));

% Loss 
        loss = mse_loss + 0.1 * peak_error + 0.2 * corr_loss;

catch ME
        fprintf('Calculation error in loss function: %s\n', ME.message);
        loss = 1e6;
end
end

%% === 模型部分 ===
function dydt = pacemaker_fhn(t,y,Z,p)
    x1=y(1); y1=y(2);
    x2=y(3); y2=y(4);
    x3=y(5); y3=y(6);
    z1=y(7); v1=y(8);
    z2=y(9); v2=y(10);
    z3=y(11); v3=y(12);
    z4=y(13); v4=y(14);
    
    a1 = p(1); a2 = p(2); a3 = p(3);
    u11 = p(4); u12 = p(5);
    u21 = p(6); u22 = p(7);
    u31 = p(8); u32 = p(9);
    f1 = p(10); f2 = p(11); f3 = p(12);
    d1 = p(13); d2 = p(14); d3 = p(15);
    e1 = p(16); e2 = p(17); e3 = p(18);
    K_SA_AV = p(19); K_AV_HP = p(20);
    tau = p(21);
    kk1 = p(22); kk2 = p(23); kk3 = p(24); kk4 = p(25);
    c1 = p(26); c2 = p(27); c3 = p(28); c4 = p(29);
    d1_ode = p(30); d2_ode = p(31); d3_ode = p(32); d4_ode = p(33);
    h1 = p(34); h2 = p(35); h3 = p(36); h4 = p(37);
    g1 = p(38); g2 = p(39); g3 = p(40); g4 = p(41);
    w11 = p(42); w12 = p(43); w21 = p(44); w22 = p(45);
    w31 = p(46); w32 = p(47); w41 = p(48); w42 = p(49);
    a_K = p(50); a_Ca = p(51); K_0 = p(52); K_1 = p(53);
    Ca_0 = p(54); Ca_1 = p(55); b1 = p(56);
    
    y2_SA_AV = Z(2);
    y3_AV_HP = Z(4);
    
    KATDe = 4e-5; KATRe = 4e-5; KVNDe = 9e-5; KVNRe = 6e-5;
    
    dydt = zeros(14,1);
    dydt(1) = y1;
    dydt(2) = -a1 * y1 * (x1 - u11) * (x1 - u12) - f1 * x1 * (x1 + d1) * (x1 + e1);
    dydt(3) = y2;
    dydt(4) = -a2 * y2 * (x2 - u21) * (x2 - u22) - f2 * x2 * (x2 + d2) * (x2 + e2) + K_SA_AV * (y2_SA_AV - y2);
    dydt(5) = y3;
    dydt(6) = -a3 * y3 * (x3 - u31) * (x3 - u32) - f3 * x3 * (x3 + d3) * (x3 + e3) + K_AV_HP * (y3_AV_HP - y3);

if dydt(1) < 0
        I_AT_De = 0;
        I_AT_Re = -KATRe * dydt(1);
else
        I_AT_De = KATDe * dydt(1);
        I_AT_Re = 0;
end

if dydt(5) < 0
        I_VN_De = 0;
        I_VN_Re = -KVNRe * dydt(5);
else
        I_VN_De = KVNDe * dydt(5);
        I_VN_Re = 0;
end

    dydt(7)  = kk1 * (-c1 * z1 * (z1 - w11) * (z1 - w12) - d1_ode * v1 * z1 + b1 * v1 + I_AT_De);
    dydt(8)  = kk1 * h1 * (z1 - g1 * v1);
    dydt(9)  = kk2 * (-c2 * z2 * (z2 - w21) * (z2 - w22) - d2_ode * v2 * z2 + I_AT_Re);
    dydt(10) = kk2 * h2 * (z2 - g2 * v2);
    dydt(11) = kk3 * (-c3 * z3 * (z3 - w31) * (z3 - w32) - d3_ode * v3 * z3 - 0.015 * v3 + I_VN_De);
    dydt(12) = kk3 * h3 * (z3 - g3 * v3);
    dydt(13) = kk4 * (-c4 * z4 * (z4 - w41 + a_K * (K_1 - K_0)) * (z4 - w42 + a_Ca * (Ca_1 - Ca_0)) - d4_ode * v4 * z4 + I_VN_Re);
    dydt(14) = kk4 * h4 * (z4 - g4 * v4);
end

function s = history(~)
    s = [0.1,0.1,0.2,0.1,0.5,0.1,0.1,0.1,0.1,0.1,0.01,0.01,0.01,0.01];
end

function [t_single, p_single] = extract_single_pwave(p, u0, tspan, fs)
if nargin < 4
        fs = 1000;
end
    tau = p(21);
try
        sol = dde23(@(t,y,Z) pacemaker_fhn(t,y,Z,p), tau, @history, tspan);
        t = linspace(tspan(1), tspan(2), round(fs*(tspan(2)-tspan(1))));
        z1 = deval(sol, t, 7);

if any(isnan(z1)) || any(isinf(z1))
            t_single = [];
            p_single = [];
return;
end

        [~, peakIdx] = max(z1);
        startIdx = peakIdx;
while startIdx > 2 && z1(startIdx) > z1(startIdx-1)
            startIdx = startIdx - 1;
end
while startIdx > 2 && z1(startIdx) < z1(startIdx-1)
            startIdx = startIdx - 1;
end

        baseline = min(z1);
        endIdx = peakIdx;
while endIdx < length(z1) - 1 && (z1(endIdx) > baseline + 1e-3 || z1(endIdx) < z1(endIdx + 1))
            endIdx = endIdx + 1;
end

        t_single = t(startIdx:endIdx) - t(startIdx);
        p_single = z1(startIdx:endIdx);

catch ME
        fprintf('Error in P wave Extraction: %s\n', ME.message);
        t_single = [];
        p_single = [];
end
end

function aligned_signal = align_peaks(model_signal, target_signal)
    [~, peak_model] = max(model_signal);
    [~, peak_target] = max(target_signal);
    shift = peak_target - peak_model;
    aligned_signal = nan(size(target_signal));

for i = 1:length(target_signal)
        src_idx = i - shift;
if src_idx >= 1 && src_idx <= length(model_signal)
            aligned_signal(i) = model_signal(src_idx);
else
            aligned_signal(i) = model_signal(end);
end
end
end