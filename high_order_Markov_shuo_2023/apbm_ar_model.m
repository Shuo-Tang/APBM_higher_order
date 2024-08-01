% APBM for auto-agressive model
% Author: Shuo Tang
% Refactoring (taming the beast) by Ondrej

clear;
close all;
rng(2);
EXPORT_GRAPHICS = false; % true if all plots should be exported
SHOW_PLOTS = true; % show plots with performance evaluation

%% Simulation Setting
Nruns = 200;%64;
Nt = 500;                      % number of iterations

%filters with learning
filters.model = {};
filters.filter = {};
filters.step = {};
filters.name = {};

% commnon settings
x_0 = [10;10];
P_0 = 0.01*eye(2);
P_1 = 0.01*eye(2);
P_2 = 0.01*eye(2);

Q = blkdiag(0.01^2*eye(2), zeros(2,2), zeros(2,2));
R = 0.1*eye(2);

% Define the AR(3) model coefficients
L = 3;
phi = [0.5, -0.3, 0.2, 0.1, -0.1, 0.05;
    0.4, -0.2, 0.1, 0.2, -0.05, 0.1]*2;
F1 = [phi(1,1), phi(1,2);phi(2,1),phi(2,2)];
F2 = [phi(1,3), phi(1,4);phi(2,3),phi(2,4)];
F3 = [phi(1,5), phi(1,6);phi(2,5),phi(2,6)];
% ╭───────────────────────────────────────────────────────────╮
% │               Setting up models and filters               │
% ╰───────────────────────────────────────────────────────────╯

% ======================================================================
% === CKF True Model (to generate data and true-model CKF)==============
% ======================================================================
m.nx = 2; % state dimension
m.anx = 6;
m.ny = 2; % measurement dimension
p = 3; % order of AR model
x0 = [zeros(2,1);zeros(2,1);x_0];
m.x0 = x0;
P0 = blkdiag(P_0, P_1, P_2);
m.P0 = P0;
m.ffun = @(x) ar3_2d(phi,x);
m.Q = Q(1:2,1:2);
m.aQ = Q;
m.hfun = @hfun;
m.R = R;
% the filter for TM is assumed to be 1st and always present
% filters = add_filter(filters,m,@ckf_tm_initialize,@ckf_tm_step,'TM_AR3');
system = m; % store model for generating data
clear m % to ensure nothing is preserved from previous model

% ====================================================
% ============== CKF PBM (1st order AR) ==============
% ====================================================
m.nx = 2; % PV state dimension
m.ny = system.ny; % P measurement dimension
m.x0 = x0(1:2);
m.P0 = P0(1:2,1:2);
m.ffun = @(x) F1*x;
m.Q = Q(1:2,1:2);
m.hfun = @hfun;
m.R = R;
% comment the following line to disable the filter
% filters = add_filter(filters,m,@pbm_initialize,@pbm_step,'PBM_AR1');
clear m % to ensure nothing is preserved from previous model

% ============================================================
% ============== APBM (1st order) ==============
% ============================================================
% m.nx = system.nx;                          % one x state dimension
% m.ny = system.ny;                          % P measurement dimension
% m.l = 1;                                   % order of model
% m.nxax = m.l*m.nx;                         % x state dimension
% m.NN = tmlp(m.nx, m.nx, 5);
% theta = m.NN.get_params();                 % getting NN parameters
% m.nnn = length(theta);
% m.nxa = m.l*m.nx + m.nnn;                  % augmented state dimension (augmented with NN parameters)
% m.x0 = [x0(1:2);zeros((m.l-1)*m.nx,1); theta];
% m.P0 = blkdiag(zeros((m.l-1)*m.nx, (m.l-1)*m.nx),P0(1:2, 1:2),1e-2*eye(m.nnn));
% m.F = F1;                                  %required for transition function
% m.ffun = @apbm_2nd_transition_function;
% m.Q = blkdiag(Q(1:2,1:2),zeros((m.l-1)*m.nx, (m.l-1)*m.nx),1e-6*eye(m.nnn));
% m.ny_pseudo = m.nnn;                       % dimension of pseudomeasurement
% m.nya = m.ny + m.ny_pseudo;                % augmented measurement dimension
% m.hfun = @apbm_nnpar_measurement_function; % controlling by NN parameter
% lambda_apbm = 0.05;
% m.R = blkdiag(R,(1/lambda_apbm)*eye(m.ny_pseudo));
% m.y_pseudo = zeros(m.ny_pseudo,1);         % theta_bar
% % comment the following line to disable the filter
% % filters = add_filter(filters,m,@apbm_2nd_nnpar_initialize,@apbm_2nd_nnpar_step,'APBM_1st');
% clear m % to ensure nothing is preserved from previous model

% ============================================================
% ============== APBM (2nd order) ==============
% ============================================================
% m.nx = system.nx;                          % PV state dimension
% m.ny = system.ny;                          % P measurement dimension
% m.l = 2;                                   % order of model
% m.nxax = m.l*m.nx;                           % x state dimension
% m.NN = tmlp(m.l*m.nx,m.nx, 5);
% theta = m.NN.get_params();                 % getting NN parameters
% m.nnn = length(theta);
% m.nxa = m.l*m.nx + m.nnn;                      % augmented state dimension (augmented with NN parameters)
% m.x0 = [x0(1:2);zeros((m.l-1)*m.nx,1); theta];
% m.P0 = blkdiag(zeros((m.l-1)*m.nx, (m.l-1)*m.nx),P0(1:2, 1:2),1e-2*eye(m.nnn));
% m.F = F1;                                  %required for transition function
% m.ffun = @apbm_2nd_transition_function;
% m.Q = blkdiag(Q(1:2,1:2),zeros((m.l-1)*m.nx, (m.l-1)*m.nx),1e-6*eye(m.nnn));
% m.ny_pseudo = m.nnn;                       % dimension of pseudomeasurement
% m.nya = m.ny + m.ny_pseudo;                % augmented measurement dimension
% m.hfun = @apbm_2nd_nnpar_measurement_function; % controlling by NN parameter
% lambda_apbm = 0.05;
% m.R = blkdiag(R,(1/lambda_apbm)*eye(m.ny_pseudo));
% m.y_pseudo = zeros(m.ny_pseudo,1);         % theta_bar
% % comment the following line to disable the filter
% % filters = add_filter(filters,m,@apbm_2nd_nnpar_initialize,@apbm_2nd_nnpar_step,'APBM_2nd');
% clear m % to ensure nothing is preserved from previous model

% ============================================================
% ============== AG-APBM (l-th order) ==============
% ============================================================
for l = 3
    m.nx = system.nx;                          % PV state dimension
    m.ny = system.ny;                          % P measurement dimension
    m.l = l;                                   % order of model
    m.nxax = m.l*m.nx;                           % x state dimension
    m.NN = tmlp(m.l*m.nx,m.nx, 5);
    theta = m.NN.get_params();                 % getting NN parameters
    m.nnn = length(theta);
    m.nxa = m.l*m.nx + m.nnn;                  % augmented state dimension (augmented with NN parameters)
    m.x0 = [x0(1:2);zeros((m.l-1)*m.nx,1); theta];
    m.P0 = blkdiag(zeros((m.l-1)*m.nx, (m.l-1)*m.nx),P0(1:2, 1:2),1e-2*eye(m.nnn));
    m.F = F1;                                  %required for transition function
    m.ffun = @apbm_ag_transition_function;
    m.Q = blkdiag(Q(1:2,1:2),0.01*ones((m.l-1)*m.nx, (m.l-1)*m.nx),1e-6*eye(m.nnn));
    m.ny_pseudo = m.nnn;                       % dimension of pseudomeasurement
    m.nya = m.ny + m.ny_pseudo;                % augmented measurement dimension
    m.hfun = @apbm_ag_nnpar_measurement_function; % controlling by NN parameter
    lambda_apbm = 0.05;
    m.R = blkdiag(R,(1/lambda_apbm)*eye(m.ny_pseudo));
    m.y_pseudo = zeros(m.ny_pseudo,1);         % theta_bar
    % comment the following line to disable the filter
    filter_name = sprintf("AG-APBM-%d", l);
    % filters = add_filter(filters,m,@apbm_ag_nnpar_initialize,@apbm_ag_nnpar_step,filter_name);
    clear m % to ensure nothing is preserved from previous model
end

% ============================================================
% ============== AP-APBM (l-th order) ==============
% ============================================================
for l = 3
    m.nx = system.nx;                          % PV state dimension
    m.ny = system.ny;                          % P measurement dimension
    m.l = l;                                   % order of model
    m.nxax = m.nx;                           % x state dimension
    m.NN = tmlp(m.l*m.nx,m.nx, 5);
    theta = m.NN.get_params();                 % getting NN parameters
    m.nnn = length(theta);
    m.nxa = m.nx + m.nnn;                      % augmented state dimension (augmented with NN parameters)
    m.x0 = [x0(1:2); theta];
    m.P0 = blkdiag(P0(1:2, 1:2),1e-2*eye(m.nnn));
    m.F = F1;                                  %required for transition function
    m.ffun = @apbm_ap_transition_function;
    m.Q = blkdiag(Q(1:2,1:2),1e-6*eye(m.nnn));
    m.ny_pseudo = m.nnn;                       % dimension of pseudomeasurement
    m.nya = m.ny + m.ny_pseudo;                % augmented measurement dimension
    m.hfun = @apbm_ap_nnpar_measurement_function; % controlling by NN parameter
    lambda_apbm = 0.05;
    m.R = blkdiag(R,(1/lambda_apbm)*eye(m.ny_pseudo));
    m.y_pseudo = zeros(m.ny_pseudo,1);         % theta_bar
    % comment the following line to disable the filter
    filter_name = sprintf("AP-APBM-%d", l);
    filters = add_filter(filters,m,@apbm_ap_nnpar_initialize,@apbm_ap_nnpar_step,filter_name);
    clear m % to ensure nothing is preserved from previous model
end


%% Simulation Running
% ╭───────────────────────────────────────────────────────────╮
% │                      RUN SIMULATIONS                      │
% ╰───────────────────────────────────────────────────────────╯

% filters with full-time learning phase
Nfilters = length(filters.filter);


% filters with full-time learning phase
save_rmse = cell(1,Nfilters);
save_anees = cell(1,Nfilters);
save_mse_cdfplot = cell(1,Nfilters);
count_id = cell(1,Nfilters);
for i = 1:Nfilters
    save_rmse{i} = NaN(Nruns,1);
    save_anees{i} = NaN(Nruns,1);
    save_mse_cdfplot{i} = zeros(Nt,Nruns);
    count_id{i} = true(1, Nruns);
end

%% Monte Carlo Experiments Repeat
tic
timeData = zeros(Nruns, 1);
memoryData = zeros(Nruns, 1);
for r = 1:Nruns
    fprintf("Monte Carlo No.%d \n", r)
    %% save computation cost
    % Record the current time and memory usage
    timeData(r) = toc;
    memInfo = memory;
    memoryData(r) = memInfo.MemUsedMATLAB;
    %% Save system Variables
    % save true state and measurement
    save_x = zeros(Nt,2);
    save_y = zeros(Nt,2);

    % save state estimates
    save_xhat = cell(1,Nfilters);
    save_xhat_anees = cell(1,Nfilters);
    for i =1:Nfilters
        save_xhat{i} = zeros(Nt,2);
        save_xhat_anees{i} = zeros(Nt,1);
    end

    % initialize filters with full-time learning phase
    for i = 1:Nfilters
        filters.filter{i}.State = filters.model{i}.x0;
        filters.filter{i}.StateCovariance = filters.model{i}.P0;
    end

    % initialize system simulator
    x = mvnrnd(x0, P0)'; % state components
    % initialize approximation for AP-APBM
    x_minus = cell(1,1);%cell(L-1,1);%
    for i = 1%:L-1 % (i+1)th-order AP-APBM
        x_minus{i} = zeros(system.nx*2, 1);
    end
    for n = 1:Nt
        %% Data Generation
        x = system.ffun(x) + mvnrnd(zeros(1,system.anx), system.aQ)';
        y = system.hfun(x) + mvnrnd(zeros(1,system.ny), R)';


        %% Save data for Analysis
        % saving true states for plotting and error computations
        save_x(n,:) = x(1:2)';
        save_y(n,:) = y';

        % run a step for each filter and store the results
        for i = 1:Nfilters
            % if i <= Nfilters - (L-1) || Nfilters <= 5
            %     [xCorr,pCorr,xPred,pPred] = filters.step{i}(filters.filter{i}, y, filters.model{i});
            % else
                % order_ap = 1;%i-2-L+1;
                % [xCorr,pCorr,xPred,pPred] = filters.step{i}(filters.filter{i}, y,...
                %     filters.model{i}, x_minus{order_ap-1});
                % x_minus{order_ap-1} = [xCorr(1:2); x_minus{order_ap-1}(1:system.nx*(order_ap-2))];
            [xCorr,pCorr,xPred,pPred] = filters.step{i}(filters.filter{i}, y,...
            filters.model{i}, x_minus{i});
            x_minus{i} = [xCorr(1:2); x_minus{i}(1:system.nx*(1))];
            % end
            save_xhat{i}(n,:) = xCorr(1:2);
            % calculating and saving ANEES
            save_xhat_anees{i}(n,:) = (xCorr(1:2)-x(1:2))'*(pCorr(1:2,1:2)\(xCorr(1:2)-x(1:2)));
        end

    end
    % save rmse for convergent trajectories
    for i = 1:Nfilters
        save_mse_cdfplot{i}(:, r) = sum((save_xhat{i} - save_x).^2, 2);

        % ******************
        % DIVERGENCE TEST
        % ******************
        thre = 1e10;
        count_id{i}(r) = ~ any(save_mse_cdfplot{i}(:, r) > thre);
        if count_id{i}(r)
            save_rmse{i}(r) = sqrt((norm(save_xhat{i} - save_x).^2)/Nt);
            save_anees{i}(r) = mean(save_xhat_anees{i})/system.nx;
        end
    end
end
toc

if SHOW_PLOTS
    %% Plots
    fontsize=16;
    set(groot,'DefaultLineLineWidth',2');
    set(groot,'defaulttextinterpreter','latex');
    set(groot,'defaultAxesTickLabelInterpreter','none');
    set(groot,'defaultLegendInterpreter','none');

    % RMSE
    h_rmse = figure;
    boxchart([cell2mat(save_rmse)])
    ax = gca; ax.FontSize = fontsize-2;
    xticklabels([filters.name])
    ylabel('RMSE (pos) [m]')
    grid
    % ANEE
    h_anees = figure;
    boxchart([cell2mat(save_anees)])
    ax = gca; ax.FontSize = fontsize-2;
    xticklabels([filters.name])
    ylabel('ANEES')
    grid

    % Empirical CDF
    h_cdf = figure;
    allRuns = 1:1:Nruns;
    % cdfplot(sqrt(sum(save_mse_cdfplot{1}(:, allRuns(count_id{1})), 2)/sum(count_id{1})))
    cdfplot(save_rmse{1})
    hold on
    for i = 2:Nfilters
        % cdfplot(sqrt(sum(save_mse_cdfplot{i}(:, allRuns(count_id{i})), 2)/sum(count_id{i})))
        cdfplot(save_rmse{i})
    end
    legend([filters.name],'fontsize', fontsize-2, 'location','best')
    xlabel('Error (pos)', 'fontsize', fontsize)
    ylabel('CDF', 'fontsize', fontsize)
    title('')
    ax = gca; ax.FontSize = fontsize-2;

    % convergence rate
    h_conv_rate = figure;
    con_rate = NaN(Nfilters,1);
    for i = 1:Nfilters
        con_rate(i) = sum(count_id{i})/Nruns;
    end
    bar(con_rate)
    xticklabels([filters.name])
    title("Convergence Rate")
end

    % plot for paper
    % RMSE
    h_rmse = figure;
    plot_index = [1,3,5,7,2];
    save_rmse_plot = cell2mat(save_rmse);
    save_rmse_plot = save_rmse_plot(:, plot_index);
    boxchart(save_rmse_plot)
    ax = gca; ax.FontSize = fontsize-2;
    label_name = [filters.name];
    label_name = label_name(plot_index);
    xticklabels(label_name)
    ylabel('RMSE')
    grid

    % Computational Cost
    h_cost = figure;
    plot(timeData(26:end),memoryData(26:end)/(1024^2) - 5200, '-o')
    memoryCost = memoryData(26:end)/(1024^2) - 5200;
    save("memoryCost_agapbm_3rd.mat", "memoryCost")
    % save("timeCost_apapbm_3rd.mat", "timeData")

if EXPORT_GRAPHICS
    exportgraphics(h_rmse, 'figs/ctr_rmse_pos_boxplots.pdf')
    saveas(h_rmse, "figs/ctr_rmse_pos_boxplots.fig")
    exportgraphics(h_anees, 'figs/ctr_anees_boxplots.pdf')
    saveas(h_anees, "figs/ctr_anees_boxplots.fig")
    exportgraphics(h_cdf, 'figs/ctr_cdf_pos_error.pdf')
    saveas(h_cdf, "figs/error_cdfplot_pos.fig")
    exportgraphics(h_conv_rate, 'figs/convergence_rate.pdf')
    saveas(h_conv_rate, "figs/convergence_rate.fig")
end

%% self=-defined functions
% ╭───────────────────────────────────────────────────────────╮
% │                          Functions                        │
% ╰───────────────────────────────────────────────────────────╯
function [filters] = add_filter(filters,model,initialization,step,filterName)
filters.model{end+1} = model;
filters.filter{end+1} = initialization(model);
filters.step{end+1} = step;
filters.name{end+1} = filterName;
end

% ======================================================================
% === CKF True Model (to generate data and true-model CKF)==============
% ======================================================================
function x_t = ar3_2d(phi, x_minus)
% Generate a single data point for a bivariate AR(3) model
% phi - matrix of AR coefficients [2x6]
% x_minus - matrix of past values [6x1]
F1 = [phi(1,1), phi(1,2);phi(2,1),phi(2,2)];
F2 = [phi(1,3), phi(1,4);phi(2,3),phi(2,4)];
F3 = [phi(1,5), phi(1,6);phi(2,5),phi(2,6)];
% Calculate the value at time t for both X and Y

% Return as a vector
x_t = F1 * x_minus(1:2) + F2 * x_minus(3:4) + F3 * x_minus(5:6);
x_t = [x_t; x_minus(1:4)];
end

function [y] = hfun(x)
y = x(1:2);
end

function [filter] = ckf_tm_initialize(m)
if isfield(m, 'aQ')
    filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.aQ,...
        'MeasurementNoise', m.R,  'StateCovariance', m.P0);
else
    filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
        'MeasurementNoise', m.R,  'StateCovariance', m.P0);
end
end

function [xCorr,pCorr,xPred,pPred] = ckf_tm_step(filter, y, m, n)
[xPred, pPred] = predict(filter);
[xCorr, pCorr] = correct(filter, y);
end

% =====================================
% ============== CKF PBM ==============
% =====================================
function [filter] = pbm_initialize(m)
filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
    'MeasurementNoise', m.R, 'StateCovariance', m.P0);
end
function [xCorr,pCorr,xPred,pPred] = pbm_step(filter, y, m, n)
[xPred, pPred] = predict(filter);
[xCorr, pCorr] = correct(filter, y);
end
% ============================================================
% ============== APBM (1st order) ==============
% ============================================================
function [x] = apbm_transition_function(x_prev, m)
s = x_prev(1:m.nx);
theta = x_prev(m.nx+1:end);
m.NN.set_params(theta)
s = m.F*s + m.NN.forward(s);
x = [s; theta];
end
function [ya] = apbm_nnpar_measurement_function(x, m)
s = x(1:m.nx);
theta = x(m.nx+1:end);
y = hfun(s);
ya = [y; theta]; % augmented y
end

function [filter] = apbm_nnpar_initialize(m)
filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
    'MeasurementNoise', m.R, 'StateCovariance', m.P0);
end

function [xCorr,pCorr,xPred,pPred] = apbm_nnpar_step(filter,y,m,n)
[xPred, pPred] = predict(filter, m);
[xCorr, pCorr] = correct(filter, [y; m.y_pseudo], m);
end

% ============================================================
% ============== AG-APBM (lth order) ==============
% ============================================================
function [x] = apbm_ag_transition_function(x_prev, m)
s = x_prev(1:m.nxax);                           
theta = x_prev(m.nxax+1:end);
m.NN.set_params(theta)
s_k = m.F*s(1:m.nx) + m.NN.forward(s);
x = [s_k;s(1:(m.l-1)*m.nx);theta];
end
function [ya] = apbm_ag_nnpar_measurement_function(x, m)
s = x(1:m.nx);
theta = x(m.nxax+1:end);
y = hfun(s);
ya = [y; theta]; % augmented y
end

function [filter] = apbm_ag_nnpar_initialize(m)
filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
    'MeasurementNoise', m.R, 'StateCovariance', m.P0);
end

function [xCorr,pCorr,xPred,pPred] = apbm_ag_nnpar_step(filter,y,m,n)
[xPred, pPred] = predict(filter, m);
[xCorr, pCorr] = correct(filter, [y; m.y_pseudo], m);
end

% ============================================================
% ============== AP-APBM (lth order) ==============
% ============================================================
function [x] = apbm_ap_transition_function(x_prev, m, x_minus)
s = x_prev(1:m.nx);                           
theta = x_prev(m.nx+1:end);
m.NN.set_params(theta)
s_k = m.F*s + m.NN.forward([s;x_minus]);
x = [s_k;theta];
end
function [ya] = apbm_ap_nnpar_measurement_function(x, m)
s = x(1:m.nx);
theta = x(m.nx+1:end);
y = hfun(s);
ya = [y; theta]; % augmented y
end

function [filter] = apbm_ap_nnpar_initialize(m)
filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
    'MeasurementNoise', m.R, 'StateCovariance', m.P0);
end

function [xCorr,pCorr,xPred,pPred] = apbm_ap_nnpar_step(filter,y,m,x_minus)
[xPred, pPred] = predict(filter, m, x_minus);
[xCorr, pCorr] = correct(filter, [y; m.y_pseudo], m);
end
