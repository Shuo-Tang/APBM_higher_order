% Implements the APBM proposed in
% Imbiriba et. al., Hybrid Neural Network Augmented Physics-based Models
% for Nonlinear Filtering.
%
% Author: Tales Imbiriba.
% Revised by Shuo
% Refactoring (taming the beast) by Ondrej
% NNcontrol: this version controls the output of the NN to be close to 0
% 0218: fixed the measurement wrapping problem by bounding the innovation
%       choose a pseudo-measurement noise, comparable to process noise

clear;
close all;
rng(2);
EXPORT_GRAPHICS = false; % true if all plots should be exported
SHOW_PLOTS = true; % show plots with performance evaluation
ALIGN_START_TIME_FOR_COMPARISON = true; % in performance evaluation align start time for filters with online learning to the filters without learning

%% Simulation Setting
Nruns = 10;%64;
Nt = 1000;                      % number of iterations
Ts = 1;                         % sampling period
F_PBM = [1 Ts 0 0;              % PBM (constant velocity model)
0 1 0 0;
0 0 1 Ts;
0 0 0 1];

% Generating elements of the process covariance matrix Qp = Gamma*Q*Gamma'
Gamma_PBM = [Ts^2/2 0;
Ts   0;
0  Ts^2/2;
0  Ts];
q = sqrt(0.01);
Q = q^2 * eye(2);

x0 = [50,0,50,0]';          % initial state for data generation
P0 = diag([0.1, 0.01, 0.1, 0.01]);  % initial state covariance matrix

% Omega (turn rate) init and covariance
Omega_0 = 0.05*pi;
Q_Omega = 1e-4;
P_Omega_0 = 1e-2;

r = sqrt(1e-2);         % noise covariance
R = r^2 * eye(2);       % noise covariance matrix

%filters with learning
filters.model = {};
filters.filter = {};
filters.step = {};
filters.name = {};

% filters without learning
filters_NL.model = {};
filters_NL.filter = {};
filters_NL.step = {};
filters_NL.name = {};

% ╭───────────────────────────────────────────────────────────╮
% │      RULE: PV STATE ALWAYS FIRST, POSITION MEASUREMENT    │
% │                       ALWAYS FIRST!!!                     │
% ╰───────────────────────────────────────────────────────────╯

% ╭───────────────────────────────────────────────────────────╮
% │               Setting up models and filters               │
% ╰───────────────────────────────────────────────────────────╯

% ======================================================================
% === CKF True Model (to generate data and true-model CKF)==============
% ======================================================================
m.nx = 4; % PV state dimension
m.ny = 2; % P measurement dimension
m.nxa = m.nx+1; % augmented state dimension (augmented with omega)
m.x0 = [x0; Omega_0];
m.P0 = blkdiag(P0,P_Omega_0);
m.ffun = @(x) ctr_transition_function(x,Ts);
m.Q = blkdiag(Gamma_PBM*Q*Gamma_PBM', Q_Omega);
m.hfun = @hfun;
m.R = R;

% the filter for TM is assumed to be 1st and always present
filters = add_filter(filters,m,@ckf_tm_initialize,@ckf_tm_step,'TM');

system = m; % store model for generating data
clear m % to ensure nothing is preserved from previous model
% =====================================
% ============== CKF PBM ==============
% =====================================
m.nx = system.nx; % PV state dimension
m.ny = system.ny; % P measurement dimension
m.x0 = x0;
m.P0 = P0;
m.ffun = @(x) F_PBM*x;
m.Q = Gamma_PBM*Q*Gamma_PBM';
m.hfun = @hfun;
m.R = R;

% comment the following line to disable the filter
filters = add_filter(filters,m,@pbm_initialize,@pbm_step,'PBM');
clear m % to ensure nothing is preserved from previous model
% ============================================================
% ============== APBM (controlled by theta_bar) ==============
% ============================================================
m.nx = system.nx;                          % PV state dimension
m.ny = system.ny;                          % P measurement dimension
m.NN = tmlp(m.nx, m.nx, 5);
theta = m.NN.get_params();                 % getting NN parameters
m.nnn = length(theta);
m.nxa = m.nx + m.nnn;                      % augmented state dimension (augmented with NN parameters)
m.x0 = [x0; theta];
m.P0 = blkdiag(P0,1e-2*eye(m.nnn));
m.F = F_PBM;                               %required for transition function
m.ffun = @apbm_transition_function;
m.Q = blkdiag(Gamma_PBM*Q*Gamma_PBM',1e-6*eye(m.nnn));
m.ny_pseudo = m.nnn;                       % dimension of pseudomeasurement
m.nya = m.ny + m.ny_pseudo;                % augmented measurement dimension
m.hfun = @apbm_nnpar_measurement_function; % controlling by NN parameter
lambda_apbm = 0.05;
m.R = blkdiag(R,(1/lambda_apbm)*eye(m.ny_pseudo));
m.y_pseudo = zeros(m.ny_pseudo,1);         % theta_bar
%
% comment the following line to disable the filter
filters = add_filter(filters,m,@apbm_nnpar_initialize,@apbm_nnpar_step,'APBM');
clear m % to ensure nothing is preserved from previous model
% ============================================================
% ==================== APBM without learning =================
% ============================================================
m.nx = system.nx;                          % PV state dimension
m.ny = system.ny;                          % P measurement dimension
m.NN = tmlp(m.nx, m.nx, 5);
m.theta = m.NN.get_params();               % NN parameters
m.nnn = length(theta);
m.x0 = x0;                                 % not needed
m.P0 = P0;                                 % not needed
m.F = F_PBM;                               % required for transition function
m.ffun = @apbm_transition_function_nl;     % the form of the function without learning
m.Q = Gamma_PBM*Q*Gamma_PBM';              % the covariance for the stage without learning
m.hfun = @hfun;                            % the form of the function without learning
m.R = R;                                   % the covariance for the stage without learning
m.initializeWith = 'APBM';                 % name oif filter which state is used to initialize this one
%
% comment the following line to disable the filter
filters_NL = add_filter(filters_NL,m,@apbm_nl_initialize,@apbm_nl_step,'APBM-NL');
clear m % to ensure nothing is preserved from previous model

% ============================================================
% ==================== APBM without learning =================
% ============================================================
m.nx = system.nx;                          % PV state dimension
m.ny = system.ny;                          % P measurement dimension
m.NN = tmlp(m.nx, m.nx, 5);
m.theta = m.NN.get_params();               % NN parameters
m.nnn = length(theta);
m.x0 = x0;                                 % not needed
m.P0 = P0;                                 % not needed
m.F = F_PBM;                               % required for transition function
m.ffun = @apbm_transition_function_nl;     % the form of the function without learning
m.Q = Gamma_PBM*Q*Gamma_PBM';              % the covariance for the stage without learning
m.hfun = @hfun;                            % the form of the function without learning
m.R = R;                                   % the covariance for the stage without learning
m.initializeWith = 'APBM_K0';                 % name oif filter which state is used to initialize this one
%
% comment the following line to disable the filter
filters_NL = add_filter(filters_NL,m,@apbm_nl_initialize,@apbm_nl_step,'APBM_K0-NL');
clear m % to ensure nothing is preserved from previous model

% ============================================================
% === APBM (controlled by theta_bar with zeroing gain) =======
% ============================================================
m.nx = system.nx;                          % PV state dimension
m.ny = system.ny;                          % P measurement dimension
m.NN = tmlp(m.nx, m.nx, 5);
theta = m.NN.get_params();                 % getting NN parameters
m.nnn = length(theta);
m.nxa = m.nx + m.nnn;                      % augmented state dimension (augmented with NN parameters)
m.x0 = [x0; theta];
m.P0 = blkdiag(1e-2*eye(m.nx),1e-2*eye(m.nnn));
m.F = F_PBM;
m.ffun = @apbm_transition_function;
m.Q = blkdiag(Gamma_PBM*Q*Gamma_PBM',1e-6*eye(m.nnn));
m.ny_pseudo = m.nnn;                       % dimension of pseudomeasurement
m.nya = m.ny + m.ny_pseudo;                % augmented measurement dimension
m.hfun = @apbm_nnpar_measurement_function; % controlling by NN parameter
lambda_apbm = 0.05;
m.R = blkdiag(R,(1/lambda_apbm)*eye(m.ny_pseudo));
m.y_pseudo = zeros(m.ny_pseudo,1);         % theta_bar

% comment the following line to disable the filter
filters = add_filter(filters,m,@apbm_nnpar_initialize,@apbm_nnpar_k0_step,'APBM_K0');
clear m % to ensure nothing is preserved from previous model
% ============================================================
% ============== APBM (controlled by NN output) ==============
% ============================================================
m.nx = system.nx;                        % PV state dimension
m.ny = system.ny;                        % P measurement dimension
m.NN = tmlp(m.nx, m.nx, 5);
theta = m.NN.get_params();               % getting NN parameters
m.nnn = length(theta);
m.nxa = m.nx + m.nnn;                    % augmented state dimension (augmented with NN parameters)
m.x0 = [x0; theta];
m.P0 = blkdiag(1e-2*eye(m.nx),1e-2*eye(m.nnn));
m.F = F_PBM;
m.ffun = @apbm_transition_function;
m.Q = blkdiag(Gamma_PBM*Q*Gamma_PBM',1e-6*eye(m.nnn));
m.ny_pseudo = m.nx;                      % dimension of pseudomeasurement
m.nya = m.ny + m.ny_pseudo;              % augmented measurement dimension
m.hfun = @apbm_nno_measurement_function; % controlling by NN output
m.R = blkdiag(R,1e-5*eye(m.ny_pseudo));
m.y_pseudo = zeros(m.ny_pseudo,1);

% comment the following line to disable the filter
filters = add_filter(filters,m,@apbm_nno_initialize,@apbm_nno_step,'APBM_NNO');
clear m % to ensure nothing is preserved from previous model
% ============================================================
% ======= APBM (controlled by NN output with zeroing gain) ===
% ============================================================
m.nx = system.nx;                        % PV state dimension
m.ny = system.ny;                        % P measurement dimension
m.NN = tmlp(m.nx, m.nx, 5);
theta = m.NN.get_params();               % getting NN parameters
m.nnn = length(theta);
m.nxa = m.nx + m.nnn;                    % augmented state dimension (augmented with NN parameters)
m.x0 = [x0; theta];
m.P0 = blkdiag(1e-2*eye(m.nx),1e-2*eye(m.nnn));
m.F = F_PBM;
m.ffun = @apbm_transition_function;
m.Q = blkdiag(Gamma_PBM*Q*Gamma_PBM',1e-6*eye(m.nnn));
m.ny_pseudo = m.nx;                      % dimension of pseudomeasurement
m.nya = m.ny + m.ny_pseudo;              % augmented measurement dimension
m.hfun = @apbm_nno_measurement_function; % controlling by NN output
eta = 1e-2;                              % multiplier of lambda0
m.R = blkdiag(R,1e-2*eye(m.ny_pseudo));
m.y_pseudo = zeros(m.ny_pseudo,1);

% comment the following line to disable the filter
filters = add_filter(filters,m,@apbm_nno_initialize,@apbm_nno_k0_step,'APBM_NNO_K0');
clear m % to ensure nothing is preserved from previous model
% ============================================================
% === APBM (controlled by NN output using multiple points) ===
% ============================================================
m.nx = system.nx;                           % PV state dimension
m.ny = system.ny;                           % P measurement dimension
m.NN = tmlp(m.nx, m.nx, 5);
theta = m.NN.get_params();                  % getting NN parameters
m.nnn = length(theta);
m.nxa = m.nx + m.nnn;                       % augmented state dimension (augmented with NN parameters)
m.x0 = [x0; theta];
m.P0 = blkdiag(1e-2*eye(m.nx),1e-2*eye(m.nnn));
m.F = F_PBM;
m.ffun = @apbm_transition_function;
m.Q = blkdiag(Gamma_PBM*Q*Gamma_PBM',1e-6*eye(m.nnn));
m.ny_pseudo = m.nx*(2*m.nx+1);              % dimension of pseudomeasurement
m.nya = m.ny + m.ny_pseudo;                 % augmented measurement dimension
m.hfun = @apbm_nno_mp_measurement_function; % controlling by NN output
m.R = blkdiag(R,1e-5*eye(m.ny_pseudo));
m.y_pseudo = zeros(m.ny_pseudo,1);

% comment the following line to disable the filter
filters = add_filter(filters,m,@apbm_nno_initialize,@apbm_nno_mp_step,'APBM_NNO_MP');
clear m % to ensure nothing is preserved from previous model


% ╭───────────────────────────────────────────────────────────╮
 % │                      RUN SIMULATIONS                      │
 % ╰───────────────────────────────────────────────────────────╯

% filters with full-time learning phase
Nfilters = length(filters.filter);

% filters without leaning phase
n_init_NL = 500; % time step to start filters without learning
Nfilters_NL = length(filters_NL.filter);

if ALIGN_START_TIME_FOR_COMPARISON
  n_init = n_init_NL;
else
  n_init = 1;
end

% filters with full-time learning phase
save_rmse_pos = cell(1,Nfilters);
save_rmse_vel = cell(1,Nfilters);
save_anees = cell(1,Nfilters);
save_mse_pos_cdfplot = cell(1,Nfilters);
save_mse_vel_cdfplot = cell(1,Nfilters);
count_id = cell(1,Nfilters);
for i = 1:Nfilters
  save_rmse_pos{i} = NaN(Nruns,1);
  save_rmse_vel{i} = NaN(Nruns,1);
  save_anees{i} = NaN(Nruns,1);
  save_mse_pos_cdfplot{i} = zeros(Nt-n_init+1,Nruns);
  save_mse_vel_cdfplot{i} = zeros(Nt-n_init+1,Nruns);
  count_id{i} = true(1, Nruns);
end

% filters without leaning phase
save_rmse_pos_NL = cell(1,Nfilters_NL);
save_rmse_vel_NL = cell(1,Nfilters_NL);
save_anees_NL = cell(1,Nfilters_NL);
save_mse_pos_cdfplot_NL = cell(1,Nfilters_NL);
save_mse_vel_cdfplot_NL = cell(1,Nfilters_NL);
count_id = cell(1,Nfilters_NL);
for i = 1:Nfilters_NL
  save_rmse_pos_NL{i} = NaN(Nruns,1);
  save_rmse_vel_NL{i} = NaN(Nruns,1);
  save_anees_NL{i} = NaN(Nruns,1);
  save_mse_pos_cdfplot_NL{i} = zeros(Nt-n_init_NL+1,Nruns);
  save_mse_vel_cdfplot_NL{i} = zeros(Nt-n_init_NL+1,Nruns);
  count_id_NL{i} = true(1, Nruns);
end


% parameters for unsecented transform
% alpha = 1e-3;
% kappa = 0;
% beta = 2;
% iota = alpha^2*(x_dim + kappa) - x_dim;
% c = x_dim + iota;


%% Monte Carlo Experiments Repeat
for r=1:Nruns
  r

  %% Save system Variables
  save_x_pos = zeros(Nt,2);
  save_x_vel = zeros(Nt,2);
  save_Omega = zeros(Nt,1);
  save_y = zeros(Nt,2);

  % Save true model turn rate
  save_xhat_Omega = zeros(Nt,1);

  % save state estimates
  save_xhat_pos = cell(1,Nfilters);
  save_xhat_vel = cell(1,Nfilters);
  save_xhat_anees = cell(1,Nfilters);
  for i =1:Nfilters
    save_xhat_pos{i} = zeros(Nt,2);
    save_xhat_vel{i} = zeros(Nt,2);
    save_xhat_anees{i} = zeros(Nt,1);
  end
  % save state estimates
  save_xhat_pos_NL = cell(1,Nfilters_NL);
  save_xhat_vel_NL = cell(1,Nfilters_NL);
  save_xhat_anees_NL = cell(1,Nfilters_NL);
  for i =1:Nfilters
    save_xhat_pos_NL{i} = zeros(Nt-n_init_NL+1,2);
    save_xhat_vel_NL{i} = zeros(Nt-n_init_NL+1,2);
    save_xhat_anees_NL{i} = zeros(Nt-n_init_NL+1,1);
  end

  % initialize filters with full-time learning phase
  for i = 1:Nfilters
    filters.filter{i}.State = filters.model{i}.x0;
    filters.filter{i}.StateCovariance = filters.model{i}.P0;
  end

  % initialize system simulator
  xPV = mvnrnd(x0, P0)'; % position velocity state components
  Omega = mvnrnd(Omega_0, P_Omega_0)';
  for n=1:Nt
    %% Data Generation
    x = system.ffun([xPV;Omega]) + mvnrnd(zeros(1,system.nxa), system.Q)';
    xPV = x(1:system.nx);
    Omega = x(end);
    y = system.hfun(xPV) + mvnrnd(zeros(1,system.ny), R)';


    %% Save data for Analysis
    % saving true states for plotting and error computations
    save_x_pos(n,:) = [xPV(1),  xPV(3)]';
    save_x_vel(n,:) = [xPV(2),  xPV(4)]';
    save_Omega(n,:) = Omega;
    save_y(n,:) = y';

    % run a step for each filter and store the results
    for i = 1:Nfilters
      [xCorr,pCorr,xPred,pPred] = filters.step{i}(filters.filter{i}, y, filters.model{i}, n);
      save_xhat_pos{i}(n,:) = [xCorr(1); xCorr(3)];
      save_xhat_vel{i}(n,:) = [xCorr(2); xCorr(4)];
      % if i == 1
      %   save_xhat_Omega(n,:) = xCorr(5);
      % end
      % calculating and saving ANEES
      save_xhat_anees{i}(n,:) = (xCorr(1:4)-xPV)'*(pCorr(1:4,1:4)\(xCorr(1:4)-xPV));%(ckf_xCorr-x)'*inv(ckf_pCorr)*(ckf_xCorr-x);
    end
    % initialize filters without learning phase
    if n == n_init_NL
      for i = 1:Nfilters_NL
        % find index of the filter used for initialization
        idx_init = find(ismember(filters.name, filters_NL.model{i}.initializeWith));
        if isempty(idx_init)
          error('Cannot find the filter for initialization');
        else
          nx = filters.model{idx_init}.nx;
          filters_NL.model{i}.theta = filters.filter{idx_init}.State(nx+1:end);
          xCorr = filters.filter{idx_init}.State(1:nx);
          pCorr = filters.filter{idx_init}.StateCovariance(1:nx,1:nx);
          filters_NL.filter{i}.State;
          filters_NL.filter{i}.StateCovariance = pCorr;
          save_xhat_pos_NL{i}(n-n_init_NL+1,:) = [xCorr(1); xCorr(3)];
          save_xhat_vel_NL{i}(n-n_init_NL+1,:) = [xCorr(2); xCorr(4)];
          % calculating and saving ANEES
          save_xhat_anees_NL{i}(n-n_init_NL+1,:) = (xCorr(1:4)-xPV)'*(pCorr(1:4,1:4)\(xCorr(1:4)-xPV));%(ckf_xCorr-x)'*inv(ckf_pCorr)*(ckf_xCorr-x);
        end
      end
    end
    if n > n_init_NL
      for i = 1:Nfilters_NL
        [xCorr,pCorr,xPred,pPred] = filters_NL.step{i}(filters_NL.filter{i}, y, filters_NL.model{i}, n);
        save_xhat_pos_NL{i}(n-n_init_NL+1,:) = [xCorr(1); xCorr(3)];
        save_xhat_vel_NL{i}(n-n_init_NL+1,:) = [xCorr(2); xCorr(4)];
        % calculating and saving ANEES
        save_xhat_anees_NL{i}(n-n_init_NL+1,:) = (xCorr(1:4)-xPV)'*(pCorr(1:4,1:4)\(xCorr(1:4)-xPV));%(ckf_xCorr-x)'*inv(ckf_pCorr)*(ckf_xCorr-x);
      end
    end
  end
  % DBG:
  % htmp = figure;
  % subplot(2,3,1)
  % plot([1:Nt],save_x_pos(:,1),'r',[1:Nt],save_xhat_pos{1}(:,1),'b')
  % title('Pos 1')
  % subplot(2,3,2)
  % plot([1:Nt],save_x_pos(:,2),'r',[1:Nt],save_xhat_pos{1}(:,2),'b')
  % title('Pos 2')
  % subplot(2,3,3)
  % plot([1:Nt],save_x_vel(:,1),'r',[1:Nt],save_xhat_vel{1}(:,1),'b')
  % title('Vel 1')
  % subplot(2,3,4)
  % plot([1:Nt],save_x_vel(:,2),'r',[1:Nt],save_xhat_vel{1}(:,2),'b')
  % title('Vel 2')
  % subplot(2,3,5)
  % plot([1:Nt],save_Omega,'r',[1:Nt],save_xhat_Omega,'b')
  % title('Omega')

  % save rmse for position and velocity only for convergent trajectories
  for i = 1:Nfilters
    save_mse_pos_cdfplot{i}(:, r) = sum((save_xhat_pos{i}(n_init:end,:) - save_x_pos(n_init:end,:)).^2,2);
    save_mse_vel_cdfplot{i}(:, r) = sum((save_xhat_vel{i}(n_init:end,:) - save_x_vel(n_init:end,:)).^2,2);

    % ******
    % DIVERGENCE TEST ON THE BASIS OF POSITION ESTIMATES ONLY
    % ******
    thre = 1e10;
    count_id{i}(r) = ~ any(save_mse_pos_cdfplot{i}(:, r) > thre);
    if count_id{i}(r)
      save_rmse_pos{i}(r) = sqrt((norm(save_xhat_pos{i}(n_init:end,:) - save_x_pos(n_init:end,:)).^2)/(Nt-n_init+1));
      save_rmse_vel{i}(r) = sqrt((norm(save_xhat_vel{i}(n_init:end,:) - save_x_vel(n_init:end,:)).^2)/(Nt-n_init+1));
      save_anees{i}(r) = mean(save_xhat_anees{i}(n_init:end))/system.nx;
    end
  end
  if n >= n_init_NL
    for i = 1:Nfilters_NL
      save_mse_pos_cdfplot_NL{i}(:, r) = sum((save_xhat_pos_NL{i} - save_x_pos(n_init_NL:end,:)).^2,2);
      save_mse_vel_cdfplot_NL{i}(:, r) = sum((save_xhat_vel_NL{i} - save_x_vel(n_init_NL:end,:)).^2,2);

      % ******
    % DIVERGENCE TEST ON THE BASIS OF POSITION ESTIMATES ONLY
    % ******
      thre = 1e10;
      count_id_NL{i}(r) = ~ any(save_mse_pos_cdfplot_NL{i}(:, r) > thre);
      if count_id_NL{i}(r)
        save_rmse_pos_NL{i}(r) = sqrt((norm(save_xhat_pos_NL{i} - save_x_pos(n_init_NL:end,:)).^2)/(Nt-n_init_NL+1));
        save_rmse_vel_NL{i}(r) = sqrt((norm(save_xhat_vel_NL{i} - save_x_vel(n_init_NL:end,:)).^2)/(Nt-n_init_NL+1));
        save_anees_NL{i}(r) = mean(save_xhat_anees_NL{i})/system.nx;
      end
    end
  end
end

if SHOW_PLOTS
  %% Plots
  fontsize=16;
  set(groot,'DefaultLineLineWidth',2');
  set(groot,'defaulttextinterpreter','latex');
  set(groot,'defaultAxesTickLabelInterpreter','none');
  set(groot,'defaultLegendInterpreter','none');

  %
  h_rmse_pos = figure;
  % A(A>thre) = NaN;
  boxchart([cell2mat(save_rmse_pos) cell2mat(save_rmse_pos_NL)])
  ax = gca; ax.FontSize = fontsize-2;
  xticklabels([filters.name filters_NL.name])
  ylabel('RMSE (pos) [m]')
  grid
  %
  h_rmse_vel = figure;
  % A(A>thre) = NaN;
  boxchart([cell2mat(save_rmse_vel) cell2mat(save_rmse_vel_NL)])
  ax = gca; ax.FontSize = fontsize-2;
  xticklabels([filters.name filters_NL.name])
  ylabel('RMSE (vel) [m]')
  grid
  %
  h_anees = figure;
  % A(A>thre) = NaN;
  boxchart([cell2mat(save_anees) cell2mat(save_anees_NL)])
  ax = gca; ax.FontSize = fontsize-2;
  xticklabels([filters.name filters_NL.name])
  ylabel('ANEES')
  grid

  % h2 = figure;
% % boxchart(A(~isnan(save_apbm_rmse), :))
% boxchart(A)
% ylim([0,100])
% ax = gca; ax.FontSize = fontsize-2;
% xticklabels({'TM','APBM','APBM-2SP','NN','PBM'})
% ylabel('RMSE [m]')
% grid
% exportgraphics(h2, 'figs/ctr_rmse_boxplots_zoom.pdf')
% saveas(h2, "figs/ctr_rmse_boxplots_zoom.fig")
%

  % tvec = [0:Nt-1]*Ts;
% h3=figure;
% plot(tvec,sum(sqrt((save_tm_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',1), hold on, grid
% plot(tvec,sum(sqrt((save_apbm_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',1)
% plot(tvec,sum(sqrt((save_apbm2sp_x_mmse - save_x).^2), 2),'-','LineWidth',1)
% plot(tvec,sum(sqrt((save_nn_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',1)
% plot(tvec,sum(sqrt((save_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',1)
% xlabel('time [s]', 'fontsize', fontsize),
% ylabel('RMSE [m]', 'fontsize', fontsize)
% legend('TM', 'APBM','APBM-2SP', 'NN', 'PBM', 'fontsize', fontsize-2, 'location','best')
% ax = gca; ax.FontSize = fontsize-2;
% exportgraphics(h3, 'figs/ctr_time_rmse.pdf')
% saveas(h3, "figs/ctr_time_rmse.fig")
%
% h4 = figure;
% plot(tvec, save_apbm_params), grid
% xlabel('time [s]', 'fontsize', fontsize)
% ylabel('\boldmath$\theta$', 'fontsize', fontsize)
% ax = gca; ax.FontSize = fontsize-2;
% exportgraphics(h4, 'figs/ctr_param_evolution.pdf')

  %
% h5 = figure;
% cdfplot(sum((save_tm_ckf_x_mmse - save_x).^2,2))
% hold on
% cdfplot(sum((save_apbm_ckf_x_mmse - save_x).^2,2))
% cdfplot(sum((save_nn_ckf_x_mmse - save_x).^2,2))
% cdfplot(sum((save_ckf_x_mmse - save_x).^2,2))
% legend('TM','APBM','NN','PBM','fontsize', fontsize-2, 'location','best')
% xlabel('squared error', 'fontsize', fontsize)
% ylabel('CDF', 'fontsize', fontsize)
% title('')
% xlim([0,4000])
% ylim([0.7,1.0])
% ax = gca; ax.FontSize = fontsize-2;
% exportgraphics(h5, 'figs/ctr_cdf_squared_error.pdf')

  h_cdf_pos = figure;
  allRuns = 1:1:Nruns;
  cdfplot(sqrt(sum(save_mse_pos_cdfplot{1}(:, allRuns(count_id{1})), 2)/sum(count_id{1})))
  hold on
  for i = 2:Nfilters
    cdfplot(sqrt(sum(save_mse_pos_cdfplot{i}(:, allRuns(count_id{i})), 2)/sum(count_id{i})))
  end
  for i = 1:Nfilters_NL
    cdfplot(sqrt(sum(save_mse_pos_cdfplot_NL{i}(:, allRuns(count_id_NL{i})), 2)/sum(count_id_NL{i})))
  end
  legend([filters.name filters_NL.name],'fontsize', fontsize-2, 'location','best')
  xlabel('Error (pos)', 'fontsize', fontsize)
  ylabel('CDF', 'fontsize', fontsize)
  title('')
  % xlim([0,4000])
% ylim([0.7,1.0])
  ax = gca; ax.FontSize = fontsize-2;

  h_cdf_vel = figure;
  allRuns = 1:1:Nruns;
  cdfplot(sqrt(sum(save_mse_vel_cdfplot{1}(:, allRuns(count_id{1})), 2)/sum(count_id{1})))
  hold on
  for i = 2:Nfilters
    cdfplot(sqrt(sum(save_mse_vel_cdfplot{i}(:, allRuns(count_id{i})), 2)/sum(count_id{i})))
  end
  for i = 1:Nfilters_NL
    cdfplot(sqrt(sum(save_mse_vel_cdfplot_NL{i}(:, allRuns(count_id_NL{i})), 2)/sum(count_id_NL{i})))
  end
  legend([filters.name filters_NL.name],'fontsize', fontsize-2, 'location','best')
  xlabel('Error (vel)', 'fontsize', fontsize)
  ylabel('CDF', 'fontsize', fontsize)
  title('')
  % xlim([0,4000])
% ylim([0.7,1.0])
  ax = gca; ax.FontSize = fontsize-2;

  h_conv_rate = figure;
  con_rate = NaN(Nfilters+Nfilters_NL,1);
  for i = 1:Nfilters
    con_rate(i) = sum(count_id{i})/Nruns;
  end
  for i = 1:Nfilters_NL
    con_rate(Nfilters+i) = sum(count_id_NL{i})/Nruns;
  end
  bar(con_rate)
  xticklabels([filters.name filters_NL.name])
  title("Convergence Rate")
end


if EXPORT_GRAPHICS
  exportgraphics(h_rmse_pos, 'figs/ctr_rmse_pos_boxplots.pdf')
  saveas(h_rmse_pos, "figs/ctr_rmse_pos_boxplots.fig")
  exportgraphics(h_rmse_vel, 'figs/ctr_rmse_vel_boxplots.pdf')
  saveas(h_rmse_vel, "figs/ctr_rmse_vel_boxplots.fig")
  exportgraphics(h_anees, 'figs/ctr_anees_boxplots.pdf')
  saveas(h_anees, "figs/ctr_anees_boxplots.fig")
  exportgraphics(h_cdf_pos, 'figs/ctr_cdf_pos_error.pdf')
  saveas(h_cdf_pos, "figs/error_cdfplot_pos.fig")
  exportgraphics(h_cdf_vel, 'figs/ctr_cdf_error_vel.pdf')
  saveas(h_cdf_vel, "figs/error_cdfplot_vel.fig")
  exportgraphics(h_conv_rate, 'figs/convergence_rate.pdf')
  saveas(h_conv_rate, "figs/convergence_rate.fig")
end


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
function [x] = ctr_transition_function(x_prev, Ts)
  %     Omega_prev = 0.05*pi;
  xPV_prev = x_prev(1:end-1);
  Omega_prev = x_prev(end);
  OTs = Omega_prev*Ts;
  cosOTs = cos(OTs);
  sinOTs = sin(OTs);
  if abs(Omega_prev)<eps % for numerical stability for small turn rate
    B = eye(4);
  else
    B = [1, sinOTs/Omega_prev    , 0, -(1-cosOTs)/Omega_prev;
    0, cosOTs               , 0, -sinOTs;
    0, (1-cosOTs)/Omega_prev, 1, sinOTs/Omega_prev;
    0, sinOTs               , 0, cosOTs];
  end
  xPV = B*xPV_prev;
  Omega = Omega_prev;
  x = [xPV; Omega];
  %     Omega = Omega_prev;
%     x = [s; Omega];
end
function [y, bounds] = hfun(x)
  y1 = 30 - 10*log10(norm(-x(1:2:3))^2.2);
  y2 = atan2(x(3),x(1));
  % y2(y2 < 0) = 2*pi + y2(y2 < 0);
  y = [y1; y2];
  bounds = [-inf inf; -pi pi];
  % y = [(norm(-x(1:2:3))^2.2); atan2(x(3),x(1))];
% y = x(1:2:3);
end
function [filter] = ckf_tm_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R,  'StateCovariance', m.P0, HasMeasurementWrapping = true);
end
function [xCorr,pCorr,xPred,pPred] = ckf_tm_step(filter,y, m, n)
  [xPred, pPred] = predict(filter);
  [xCorr, pCorr] = correct(filter, y);
end
% =====================================
% ============== CKF PBM ==============
% =====================================
function [filter] = pbm_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R, 'StateCovariance', m.P0, HasMeasurementWrapping = true);
end
function [xCorr,pCorr,xPred,pPred] = pbm_step(filter, y, m, n)
  [xPred, pPred] = predict(filter);
  [xCorr, pCorr] = correct(filter, y);
end
% ============================================================
% ============== APBM (Controlled by theta_bar) ==============
% ============================================================
function [x] = apbm_transition_function(x_prev, m)
  s = x_prev(1:m.nx);
  theta = x_prev(m.nx+1:end);
  m.NN.set_params(theta)
  % s = w(1)*F*s + w(2)*nn_mlp.forward(s);
  s = m.F*s + m.NN.forward(s);
  % x = [theta; w; s];
  x = [s; theta];
end
function [ya, bounds] = apbm_nnpar_measurement_function(x, m)
  s = x(1:m.nx);
  theta = x(m.nx+1:end);
  % w = x(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
  [y, bounds_y] = hfun(s); %[30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
  ya = [y; theta]; % augmented y
  bounds_theta = repmat([-inf inf], m.NN.nparams, 1);
  bounds = [bounds_y;bounds_theta];
end
function [filter] = apbm_nnpar_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R, 'StateCovariance', m.P0, HasMeasurementWrapping = true);
end

function [xCorr,pCorr,xPred,pPred] = apbm_nnpar_step(filter,y,m,n)
  [xPred, pPred] = predict(filter, m);
  [xCorr, pCorr] = correct(filter, [y; m.y_pseudo], m);
end

% ============================================================
% ==================== APBM without learning =================
% ============================================================
function [x] = apbm_transition_function_nl(x_prev, m)
  m.NN.set_params(m.theta)
  x = m.F*x_prev + m.NN.forward(x_prev);
end
function [filter] = apbm_nl_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R, 'StateCovariance', m.P0, HasMeasurementWrapping = true);
end
function [xCorr,pCorr,xPred,pPred] = apbm_nl_step(filter, y, m, n)
  [xPred, pPred] = predict(filter, m);
  [xCorr, pCorr] = correct(filter, y);
end
% ============================================================
% === APBM (controlled by theta_bar with zeroing gain) =======
% ============================================================
function [xCorr,pCorr,xPred,pPred] = apbm_nnpar_k0_step(filter,y,m,n)
  [xPred, pPred] = predict(filter, m);

  kappa= 0; % sigma point parameter
  Xs = sigmas(xPred,pPred,kappa); % predictive sigma points of x
  W = [kappa 0.5*ones(1,2*m.nxa)]/(m.nxa+kappa); %  associated weights
  Ns = size(Xs,2); % # samples
  Ys = zeros(m.nya,Ns);
  for i = 1:Ns % propagate sigma points through the nonlinear function
    [Ys(:,i),~] = m.hfun(Xs(:,i),m);
  end
  Ym = Ys*W';
  Pxy = ((Xs-xPred).*W)*(Ys-Ym)';
  HX = Pxy'/pPred; % statistical linearization

  Ycov = ((Ys-Ym).*W)*(Ys-Ym)';
  Pyy = Ycov + m.R;
  % [diag(Ycov) eig(Ycov) eig(Ycov+R_apbmk0)]
  KG = Pxy/Pyy;

  % zeroing block relating the pseudomeasurement (bottom part) to the state (upper part)
  KG(1:m.nx,m.ny+1:end) = 0;

  ya = [y; m.y_pseudo]; % augmented measurement
  y_diff = (ya-Ym);
  % Wrapping angle
  if abs(y_diff(2)) < pi
    % do nothing
  elseif y_diff(2) > pi
    y_diff(2) = y_diff(2)-pi;
  elseif y_diff(2) < -pi
    y_diff(2) = y_diff(2)+pi;
  end


  xCorr = xPred + KG*y_diff;
  pCorr = (eye(m.nxa)-KG*HX)*pPred*(eye(m.nxa)-KG*HX)' + KG*m.R*KG';

  filter.State = xCorr;
  filter.StateCovariance = pCorr;
end
% ============================================================
% ============== APBM (Controlled by NN output) ==============
% ============================================================
function [ya, bounds] = apbm_nno_measurement_function(x, m,xPV_pred)
  s = x(1:m.nx);
  theta = x(m.nx+1:end);
  % w = x(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
  m.NN.set_params(theta)
  [y, bounds_y] = hfun(s); %[30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
  y_pseudo = m.NN.forward(xPV_pred);
  ya = [y; y_pseudo]; % augmented y
  bounds_y_pseudo = repmat([-inf inf], m.nx, 1);
  bounds = [bounds_y; bounds_y_pseudo];
end
function [filter] = apbm_nno_initialize(m)
  filter = trackingCKF(m.ffun, m.hfun, m.x0, 'ProcessNoise', m.Q,...
  'MeasurementNoise', m.R, 'StateCovariance', m.P0, HasMeasurementWrapping = true);
  % filter.State = m.x0;
  % filter.StateCovariance = m.P0;
end
function [xCorr,pCorr,xPred,pPred] = apbm_nno_step(filter,y,m,n)
  [xPred, pPred] = predict(filter, m);
  [xCorr, pCorr] = correct(filter, [y; m.y_pseudo], m, xPred(1:m.nx));
end
% ============================================================
% ======= APBM (controlled by NN output with zeroing gain) ===
% ============================================================
function [xCorr,pCorr,xPred,pPred] = apbm_nno_k0_step(filter,y,m,n)
  [xPred, pPred] = predict(filter, m);
  kappa= 0; % sigma point parameter
  Xs = sigmas(xPred,pPred,kappa); % predictive sigma points of x
  W = [kappa 0.5*ones(1,2*m.nxa)]/(m.nxa+kappa); %  associated weights
  Ns = size(Xs,2);
  Ys = zeros(m.nya,Ns);
  for i = 1:Ns % propagate sigma points through the nonlinear function
    [Ys(:,i),~] = apbm_nno_measurement_function(Xs(:,i),m,xPred(1:m.nx));
  end
  Ym = Ys*W';
  Pxy = ((Xs-xPred).*W)*(Ys-Ym)';
  HX = Pxy'/pPred; % statistical linearization

  Ycov = ((Ys-Ym).*W)*(Ys-Ym)';
  Pyy = Ycov +m.R;
  % [diag(Ycov) eig(Ycov) eig(Ycov+R_apbm1spk0)]
  KG = Pxy/Pyy;
  % zeroing block relating the pseudomeasurement to the state
  KG(1:m.nx,m.ny+1:end) = 0;

  ya = [y; m.y_pseudo]; % augmenting measurement and pseudo-measurement
  y_diff = (ya-Ym);
  % Wrapping angle
  if abs(y_diff(2)) < pi
    % do nothing
  elseif y_diff(2) > pi
    y_diff(2) = y_diff(2)-pi;
  elseif y_diff(2) < -pi
    y_diff(2) = y_diff(2)+pi;
  end

  xCorr = xPred + KG*y_diff;
  pCorr = (eye(m.nxa)-KG*HX)*pPred*(eye(m.nxa)-KG*HX)' + KG*m.R*KG';
  % apbm1spk0_pCorr = apbm1spk0_pPred - Pxy/Pyy*Pxy';

  filter.State = xCorr;
  filter.StateCovariance = pCorr;
end

% ============================================================
% === APBM (controlled by NN output using multiple points) ===
% ============================================================
function [ya, bounds] = apbm_nno_mp_measurement_function(x, m,xPV_pred,PPV_pred)
  s = x(1:m.nx);
  theta = x(m.nx+1:end);
  % w = x(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
  m.NN.set_params(theta)
  [y, bounds_y] = hfun(s); %[30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
  kappa = 1;
  Xs = sigmas(xPV_pred,PPV_pred,kappa);
  y_pseudo = zeros(m.ny_pseudo,1);
  for i = 1:(2*m.nx+1)
    y_pseudo([1:m.nx]+(i-1)*m.nx) = m.NN.forward(Xs(:,i));
  end
  ya = [y; y_pseudo]; % augmented y
  bounds_y_pseudo = repmat([-inf inf], m.ny_pseudo, 1);
  bounds = [bounds_y; bounds_y_pseudo];
end

function [xCorr,pCorr,xPred,pPred] = apbm_nno_mp_step(filter,y,m,n)
  [xPred, pPred] = predict(filter, m);
  %kappa = 1;
  %W = [kappa 0.5*ones(1,2*m.nx)]/(m.nx+kappa); % can be used to adjust the block corresponding to pseudomeasurement noise covariance
  [xCorr, pCorr] = correct(filter, [y; m.y_pseudo], m, xPred(1:m.nx),pPred(1:m.nx,1:m.nx));
end






function y = nn_reg_measurement_function(x, nn_mlp)
  %     global nn_mlp
  theta = x(1:nn_mlp.nparams);
  s = x(nn_mlp.nparams + 1: end);
  %     y = [30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
  y = [theta; hfun(s)]; %30 - 10*log10(norm(-s)^2.2); atan2(s(2),s(1))];
end


function y = apbm2sp_measurement_function1(x, nn_mlp)
  %   standard measurement: no controll from w & theta
    % theta = x(1:nn_mlp.nparams);
    % w = x(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
  s = x(nn_mlp.nparams + 1: end);
  y = hfun(s); % [30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
end

function y = apbm2sp_measurement_function2(x, Ts, nn_mlp)
  %   pseudo-measurement: control the output of APBM by PBM
  F = [1 Ts 0 0;
  0 1 0 0;
  0 0 1 Ts;
  0 0 0 1];
  theta = x(1:nn_mlp.nparams);
  w = x(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
  s = x(nn_mlp.nparams + 3: end);
  nn_mlp.set_params(theta)
  y = w(1)*F*s + w(2)*nn_mlp.forward(s);
end

function y = apbm2sp_NN_measurement_function(x, nn_mlp, x_est)
  %   pseudo-measurement: control the output of APBM by PBM
  theta = x(1:nn_mlp.nparams);
  % s = x(nn_mlp.nparams + 1: end);
  s = x_est;
  nn_mlp.set_params(theta)

  y = nn_mlp.forward(s);
  % y
end

function [X] = sigmas(x,P,kappa)
  %Sigma points around reference point
%Inputs:
%       x: reference point
%       P: covariance
%       kappa: parameter
%Output:
%       X: Sigma points
  nx = length(x);
  [U,S,~] = svd(P);
  sP = U*sqrt(S);
  % sP = chol(P)';
  A = sqrt(nx+kappa)*sP;
  X = x + [zeros(nx,1) A -A];
end

function [y, bounds] = apbm1sp_measurement_function_sigma(x, nn_mlp, Xs, x_dim)
  %   pseudo-measurement: control the output of APBM by PBM
  theta = x(1:nn_mlp.nparams);
  s = x(nn_mlp.nparams + 1: end);
  nn_mlp.set_params(theta)
  % real measurements
  [y1, bounds_y1] = hfun(s); %[30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
  % pseudo-measurements
  N = size(Xs,2);
  y2 = zeros(N*x_dim, 1);
  for i = 1:N
    y2(x_dim*(i-1)+1: x_dim*i) = nn_mlp.forward(Xs(:, i));
  end
  bounds_y2 = repmat([-inf inf], N*x_dim, 1);

  y = [y1;y2];
  bounds = [bounds_y1;bounds_y2];
  % y
end

