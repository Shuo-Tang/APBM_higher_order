    % Data generator and predictors (model based, AI data-based, APBM)
% - 1D position model - random walk model with measured position increment
% - for APBM concept illustration 
% - joint paper "AI-based Modelling for Navigation and Tracking"

clear; close;

% Initialisation
% - number of simulated data
N = 1e3;
% - initial position PDF
pinit_mean = 0;
pinit_var = 10^3;
% - noise properties
% -- position white noise variance
Q = 2^2;
% -- measurement noise variance
R = .1^2;
% -- position increment measurement bias driving noise variance (time constant is varying and defined later)
Sigma2 = 1e-1^2;
% -- relation between time constant and independent variable
tauFunct = @(theta) exp(-1./(1e2*(theta-14.995)));
% - allocation
% -- true position, position increment, measured increment, bias, bias time constant, independent variable (e.g., temperature)
p = zeros(1,N);
y = zeros(1,N); % noisy position measurement
u = zeros(1,N);
u_measured = zeros(1,N);
b = zeros(1,N);
tau = zeros(1,N);
theta = zeros(1,N);
theta_measured = zeros(1,N);
R_theta = 1^2;
% -- model-based predicted quantities 
p_mb1 = zeros(1,N);
b_mb1 = zeros(1,N);
p_mb2 = zeros(1,N);
p_mb3opt = zeros(1,N);
b_mb3opt = zeros(1,N);
% -- data-based predicted quantities (AI/ML/Full NN) 
p_db = zeros(1,N);
% -- APBM quantities 
p_apbm = zeros(1,N);

% Data Generator
p(1) = sqrt(pinit_var)*randn + pinit_mean;
y(1) = p(1) + sqrt(R)*randn;
for k = 1:N
    % - true position increment
    u(k) = sin(1+0.1*randn);
    % - independent variable value (e.g., temperature - between 15 and 25 degC)
    theta(k) = 20 + 5*cos(k*1e-3); % 12, 20, 120
    theta_measured(k) = theta(k) + sqrt(R_theta)*trnd(3);%sqrt(R_theta)*randn;
    %theta(k) = 20; % 12, 20, 120
    % - dependent variable valu (i.e., time constant)
    tau(k) = tauFunct(theta(k));
    % - position
    w = sqrt(Q)*randn; % position noise
    p(k+1) = p(k) + u(k) + w; % position dynamics
    y(k+1) = p(k+1) + sqrt(R)*trnd(3);
    % - bias
    xi = sqrt(Sigma2)*randn; % bias noise
    b(k+1) = tau(k)*b(k) + xi; % bias dynamics
    % - measured position increment
    u_measured(k) = u(k) + b(k);
end
% --
% disp('Variance: state noise Q, bias driving noise Sigma2, bias overall')
% disp([Q, Sigma2, var(b)])
table([Q, Sigma2, var(b)]','RowNames',{'state noise Q', 'bias driving noise Sigma2', 'bias overall'},'VariableNames',{'variance'})

% Model-based prediction (one-step)
% - mean value and covariance matrix
%meanTau = tauFunct(mean(tau));
meanTau = (max(tau)+min(tau))/2;
% -- KF related quantities
F = [1, -1; 0 meanTau];
H = [1 0];
Sigma2Overbounded = Sigma2*110;    % ***** Tunable overbounding for varying tau 
Qx = diag([Q,Sigma2Overbounded]);
% R = 0;
% Pp = diag([10, 1]);
xp = zeros(2,N);
xf = zeros(2,N);
PpSteady = dare(F',H',Qx,R);
KSteady = (PpSteady*H')/(H*PpSteady*H'+R);
PfSteady = PpSteady - KSteady*H*PpSteady;


%% Notation and basic definitions.
% The assumption is that the state vector x can be split in two parts x =
% [b; p] each one possible a multidimensional vector. 
% the different models below exploits different NN and state formulation by
% modeling the full state [b; p] or only the 'p' part. We also assume that 
% we have a input u for the dynamical model f(x,u) and that we have a
% measurement theta correlated with b. 
%
% ==========================
% Notation and basic definitions
% states x = [b; p] (bias 'b' and position 'p')
% x -> states
% p -> positions
% b -> bias
% phi -> nn parameters
% xa = [phi; x] -> augmented state vector

% augmented states xa = [nnparams; x] = [phi; b, p]

% dimention of inputs and measurements
u_dim = size(u_measured,1);
y_dim = size(y,1);
theta_dim = size(theta_measured,1);

% state part dimensions:
p_dim = size(p,1);
b_dim = size(b,1);
x_dim = p_dim + b_dim;


% defining transition functions
Fall = [1 0; 
      -1 1];
f_all = @(x, u) Fall * x + u;

% or x = [p] (position only)
f_p = @(p,u) p + u;


%%  Defining APBM and NN models 

% ====== APBM Opt 0 =======
% Models only position so
% x = p, and xa = [phi,p]
% x = f1d(x,u) + NN(u, theta_measured; phi)
% 
n_states = size(p,1);
x_dim_opt_0 = n_states;

nn_input_dim = u_dim + theta_dim; % NN(u,theta)
hidden_layers = [5];

% global nn_mlp_opt_0;              % Neural Network needed inside the 
nn_tfunc = @nn_transition_function_opt_0;             % APBM transition function
nn_hfun = @nn_reg_measurement_function_opt_0;         % regularized measurement function
% APBM initialization 
nn_mlp_opt_0 = tmlp(nn_input_dim, x_dim_opt_0, hidden_layers);     % creating NN object
nnparams = nn_mlp_opt_0.get_params();                        % getting NN parameters
x0 = zeros(x_dim_opt_0, 1);
xa = [nnparams; x0];                                  % initial NN_CKF states

% NN process noise
% Q_nn = q^2*eye(length(x_nn));                       
Q_nn = 1e-5*eye(length(xa));
% TODO: set a better Qx
Q_nn(end-x_dim_opt_0+1:end, end-x_dim_opt_0+1:end) = 0.01;

% Initial NN state cov 
P_nn = 1e-2*eye(length(xa));
P_nn(end-x_dim_opt_0+1:end, end-x_dim_opt_0+1:end) = 1e-2*eye(x_dim_opt_0);

% noise covariance matrix for augmented likelihood model (for
% regularization)
lambda = 1e-3;
R_nn = (1/lambda)*eye(length(xa));
% TODO: set a better Ry
R_nn(end-y_dim+1:end,end-y_dim+1:end) = max(R,1e-6);

nn_ckf_opt_0 = trackingCKF(nn_tfunc, nn_hfun, xa, 'ProcessNoise', Q_nn, 'MeasurementNoise', R_nn, 'StateCovariance', P_nn);

save_nn_ckf_x_opt_0 = zeros(N,1);
save_nn_params_opt_0 = zeros(N, nn_mlp_opt_0.nparams);
save_nn_output_opt_0 = zeros(N, x_dim_opt_0);
% zero vector for likelihood augmentation
zero_meas = zeros(nn_mlp_opt_0.nparams,1);


% ====== APBM Opt 2 =======
% x = [phi, b, p] = [nnparam; s]
% y = p
% \tilde(y) = [phi;y]
%
% phi = phi + q_phi
% b = NN(b, u_measured, theta; nnparams) + q_b
% p = F*p + u_measured - b + q_p;

n_states = p_dim + b_dim;
nn_output_dim_opt_2 = b_dim;
nn_input_dim_opt_2 = b_dim + u_dim + theta_dim;
x_dim_opt_2 = n_states;
hidden_layers = [5];

% APBM initialization 
nn_mlp_opt_2 = tmlp(nn_input_dim_opt_2, nn_output_dim_opt_2, hidden_layers);     % creating NN object
nnparams = nn_mlp_opt_2.get_params();                        % getting NN parameters
x0 = zeros(x_dim_opt_2, 1);
xa = [nnparams; x0];                                  % initial NN_CKF states

% NN process noise
Q_nn = 1e-5*eye(length(xa));
Q_nn(end-x_dim_opt_2+1:end, end-x_dim_opt_2+1:end) = 1e-2 * eye(x_dim_opt_2);

% Initial NN state cov 
P_nn = 1e-2*eye(length(xa));
P_nn(end-x_dim_opt_2+1:end, end-x_dim_opt_2+1:end) = 1e-2*eye(x_dim_opt_2);

% noise covariance matrix for augmented likelihood model (for
% regularization)
lambda = 1e-3;
R_nn = (1/lambda)*eye(length(xa)-1);
% TODO: set a better Ry
R_nn(end - 1, end - 1) = 1e-6;
R_nn(end-y_dim+1:end,end-y_dim+1:end) = max(1e-6, R);

nn_tfunc_opt_2 = @nn_transition_function_opt_2;             % APBM transition function
nn_hfun_opt_2 = @nn_reg_measurement_function_opt_2;         % regularized measurement function

nn_ckf_opt_2 = trackingCKF(nn_tfunc_opt_2, nn_hfun_opt_2, xa, 'ProcessNoise', Q_nn, 'MeasurementNoise', R_nn, 'StateCovariance', P_nn);

save_nn_ckf_x_opt_2 = zeros(N,1);
save_nn_params_opt_2 = zeros(N, nn_mlp_opt_2.nparams);
save_nn_output_opt_2 = zeros(N, nn_output_dim_opt_2);
% zero vector for likelihood augmentation
zero_meas_opt2 = zeros(nn_mlp_opt_2.nparams,1);



% ====== APBM Opt 3 =======
% xa = [phi_0; phi_1; nnparam; b; p] = [phi_0; phi_1; phi; x]
% y = p
% \tilde(y) = [phi_0; phi_1; phi;y]
%
% x = NN_tilde(F*x + [u_measured;0], u_measured, theta; nnparams);
%   = phi0 * (F*x) + phi1 * (NN(x, u_measured, theta; phi(2:))


nn_output_dim_opt_3 = x_dim;
nn_input_dim_opt_3 = x_dim + u_dim + theta_dim;
x_dim_opt_3 = x_dim;
hidden_layers = [5];

% global nn_mlp_opt_3;              % global Neural Network needed inside the 
% APBM initialization 
nn_mlp_opt_3 = tmlp(nn_input_dim_opt_3, nn_output_dim_opt_3, hidden_layers);     % creating NN object
nnparams = nn_mlp_opt_3.get_params();                        % getting NN parameters

x0 = zeros(x_dim_opt_3, 1);
% total model params = nnparams + 2, where +2 comes from phi_0 and phi_1
xa = [1e-1*randn + 1; 1e-1*randn; nnparams; x0];                                  % initial NN_CKF states

% NN process noise
Q_nn = 1e-5*eye(length(xa));
Q_nn(end-x_dim_opt_3+1:end, end-x_dim_opt_3+1:end) = 1e-2 * eye(x_dim_opt_3);

% Initial NN state cov 
P_nn = 1e-2*eye(length(xa));
P_nn(end-x_dim_opt_3+1:end, end-x_dim_opt_3+1:end) = 1e-2*eye(x_dim_opt_3);

% noise covariance matrix for augmented likelihood model (for
% regularization)
lambda = 1e-3;
R_nn = (1/lambda)*eye(length(xa)-1);
% TODO: set a better Ry
R_nn(end - 1, end - 1) = 1e-6;
R_nn(end-y_dim+1:end,end-y_dim+1:end) = max(1e-6, R);


nn_tfunc_opt_3 = @nn_transition_function_opt_3;             % APBM transition function
nn_hfun_opt_3 = @nn_reg_measurement_function_opt_3;         % regularized measurement function

nn_ckf_opt_3 = trackingCKF(nn_tfunc_opt_3, nn_hfun_opt_3, xa, 'ProcessNoise', Q_nn, 'MeasurementNoise', R_nn, 'StateCovariance', P_nn);

save_nn_ckf_p_opt_3 = zeros(N,p_dim);
save_nn_params_opt_3 = zeros(N, nn_mlp_opt_3.nparams + 2);
save_bias_opt_3 = zeros(N, b_dim);
% zero vector for likelihood augmentation
zero_meas_opt3 = zeros(nn_mlp_opt_3.nparams,1);
% adding regularization for phi_0=1 and phi_1=0
zero_meas_opt3 = [1; 0; zero_meas_opt3];



% ====== APBM Opt 4 =======
% x = [phi, b, p] = [nnparam; s]
% y = p
% \tilde(y) = [phi;y]
%
% phi = phi + q_phi
% p = NN(p, u_measured, theta) + q_x

nn_output_dim_opt_4 = p_dim;
nn_input_dim_opt_4 = p_dim + u_dim + theta_dim;
x_dim_opt_4 = p_dim;
hidden_layers = [5];

% global nn_mlp_opt_4;              % global Neural Network needed inside the 
% APBM initialization 
nn_mlp_opt_4 = tmlp(nn_input_dim_opt_4, nn_output_dim_opt_4, hidden_layers);     % creating NN object
nnparams = nn_mlp_opt_4.get_params();                        % getting NN parameters

x0 = zeros(x_dim_opt_4,1);
xa = [nnparams; x0];                                  % initial NN_CKF states

% NN process noise
Q_nn = 1e-5*eye(length(xa));
Q_nn(end-x_dim_opt_4+1:end, end-x_dim_opt_4+1:end) = 1e-2 * eye(x_dim_opt_4);

% Initial NN state cov 
P_nn = 1e-2*eye(length(xa));
P_nn(end-x_dim_opt_4+1:end, end-x_dim_opt_4+1:end) = 1e-2*eye(x_dim_opt_4);

% noise covariance matrix for augmented likelihood model (for
% regularization)
% lambda = 10;
% R_nn = (1/lambda)*eye(length(x_nn)-1);
% TODO: set a better Ry
% R_nn(end - 1, end - 1) = 1e-6;
% R_nn(end-y_dim+1:end,end-y_dim+1:end) = 1e-6;
R_nn = max(1e-6, R);

nn_tfunc_opt_4 = @nn_transition_function_opt_4;             % APBM transition function
nn_hfun_opt_4 = @nn_reg_measurement_function_opt_4;         % regularized measurement function

nn_ckf_opt_4 = trackingCKF(nn_tfunc_opt_4, nn_hfun_opt_4, xa, 'ProcessNoise', Q_nn, 'MeasurementNoise', R_nn, 'StateCovariance', P_nn);

save_nn_ckf_x_opt_4 = zeros(N,x_dim_opt_4);
save_nn_params_opt_4 = zeros(N, nn_mlp_opt_4.nparams);
save_nn_output_opt_4 = zeros(N, nn_output_dim_opt_4);
% zero vector for likelihood augmentation
zero_meas_opt4 = zeros(nn_mlp_opt_4.nparams,1);


for k = 1:N-2
    % ------
    % one-step (approx 1 - neglecting w(k) and b(k))
    % ------
    % - position prediction
%     p_mb1(k+1) = p(k) + u_measured(k) - b_mb1(k);
    p_mb1(k+1) = y(k) + u_measured(k) - b_mb1(k);
    % - bias prediction (zero - depending on the initial condition)
    b_mb1(k+1) = meanTau*b_mb1(k);
    % ------
    % one-step (approx 2 - neglecting w(k) and w(k-1))
    % ------
    % - position prediction
    if k>=3
%         p_mb2(k+1) = (1+meanTau)*p(k) - meanTau*p(k-1) + u_measured(k) - meanTau*u_measured(k-1);
        p_mb2(k+1) = (1+meanTau)*y(k) - meanTau*p(k-1) + u_measured(k) - meanTau*u_measured(k-1);
    end
    % ------
    % - one-step (opt KF-based - up to exactly unknown time constant)
    % ------
    %K = Pp*H'/(H*Pp*H'+R);
    zp = H*xp(:,k);
    %xf(:,k) = xp(:,k) + KSteady*(p(k)-zp);
    xf(:,k) = xp(:,k) + KSteady*(y(k)-zp);
    %Pf = Pp - K*H*Pp;
    % --
    xp(:,k+1) = F*xf(:,k) + [u_measured(k); 0];
    %Pp = F*Pf*F' + Qx;
    % --
    p_mb3opt(k+1) = xp(1,k+1);
    
    % ---- APBM opt 0 -------
    [nn_xPred, nn_pPred] = predict(nn_ckf_opt_0, u_measured(k), theta_measured(k), f_p, nn_mlp_opt_0);
    % p_k+1 = p_k + u_measured - NN(...).
    % NN(...) = p_k + u_measured - p_k+1
    % b = 
    save_nn_output_opt_0(k+1) = save_nn_ckf_x_opt_0(k) - u_measured(k) - nn_xPred(end-x_dim_opt_0+1:end);
    % correct with augmented likelihood function:
    [nn_ckf_xCorr, nn_ckf_pCorr] = correct(nn_ckf_opt_0, [zero_meas ; y(k)], nn_mlp_opt_0);     
    
    % testing/ making things flowing
%     P = nn_ckf.StateCovariance;
%     min_eig = min(eig(P));
%     if min_eig < 1e-8
% %         disp('here')
%         nn_ckf.StateCovariance = nn_ckf.StateCovariance  + 10*min_eig*eye(size(nn_ckf.StateCovariance));
%     end
    
    % getting nn params
    save_nn_params_opt_0(k+1,:) = nn_ckf_xCorr(1:end-x_dim_opt_0);
    % getting only p states (not parameters)
    nn_ckf_xCorr = nn_ckf_xCorr(end-p_dim+1:end);
    
    % saving nn_ckf estimated states
    save_nn_ckf_x_opt_0(k+1,:) = nn_ckf_xCorr;
    
    
    % ---- APBM opt 2 -------
    
    [nn_xPred, ~] = predict(nn_ckf_opt_2, u_measured(k), theta_measured(k), f_p, nn_mlp_opt_2);
    
    % correct with augmented likelihood function:
    [nn_ckf_xCorr, ~] = correct(nn_ckf_opt_2, [zero_meas_opt2 ; y(k)], nn_mlp_opt_2);     
    
    % getting nn params
    save_nn_params_opt_2(k+1,:) = nn_ckf_xCorr(1:end-x_dim_opt_2);
    % getting only states (not parameters)
    % saving bias term
    save_nn_output_opt_2(k+1) = nn_ckf_xCorr(end-x_dim+1:end-p_dim);
    
    % saving nn_ckf estimated p 
    nn_ckf_xCorr = nn_ckf_xCorr(end-p_dim+1 : end);
    save_nn_ckf_x_opt_2(k+1,:) = nn_ckf_xCorr;
    
    % ---- APBM opt 3 -------
    
    [nn_xPred, ~] = predict(nn_ckf_opt_3, u_measured(k), theta_measured(k), f_all, nn_mlp_opt_3);
    
    % correct with augmented likelihood function:
    [nn_ckf_xCorr, ~] = correct(nn_ckf_opt_3, [zero_meas_opt3 ; y(k)], nn_mlp_opt_3);     
    
    % getting nn params
    save_nn_params_opt_3(k+1,:) = nn_ckf_xCorr(1:end-x_dim_opt_3);
    % getting only states (not parameters)
    % saving bias term
    save_bias_opt_3(k+1) = nn_ckf_xCorr(end-x_dim+1:end-p_dim);
    
    % saving nn_ckf estimated states
    nn_ckf_xCorr = nn_ckf_xCorr(end-p_dim+1 : end);
    save_nn_ckf_p_opt_3(k+1,:) = nn_ckf_xCorr;
    
       
    % ---- APBM opt 4 -------
    
    [nn_xPred, ~] = predict(nn_ckf_opt_4, u_measured(k), theta_measured(k), nn_mlp_opt_4);
    save_nn_output_opt_4(k+1) = nn_xPred(end-1);
    
    % correct with augmented likelihood function:
%     [nn_ckf_xCorr, ~] = correct(nn_ckf_opt_4, [zero_meas_opt4 ; y(k)]);     
    [nn_ckf_xCorr, ~] = correct(nn_ckf_opt_4, y(k), nn_mlp_opt_4);     
    
    % getting nn params
    save_nn_params_opt_4(k+1,:) = nn_ckf_xCorr(1:end-x_dim_opt_4);
    % getting only states (not parameters)
    nn_ckf_xCorr = nn_ckf_xCorr(end);
    
    % saving nn_ckf estimated states
    save_nn_ckf_x_opt_4(k+1,:) = nn_ckf_xCorr;
    
end
%%
MSE_mb1 = var(p(1:N-1)-p_mb1(1:N-1));
MSE_mb2 = var(p(3:N-1)-p_mb2(3:N-1));
MSE_mb3opt = var(p(1:N-1)-p_mb3opt(1:N-1)); % should be same as PpSteady(1,1)
MSE_apbm_opt_0 = var(p(1:N-1)-save_nn_ckf_x_opt_0(1:N-1)'); % should be same as PpSteady(1,1)
MSE_apbm_opt_2 = var(p(1:N-1)-save_nn_ckf_x_opt_2(1:N-1)'); % should be same as PpSteady(1,1)
MSE_apbm_opt_3 = var(p(1:N-1)-save_nn_ckf_p_opt_3(1:N-1)'); % should be same as PpSteady(1,1)
MSE_apbm_opt_4 = var(p(1:N-1)-save_nn_ckf_x_opt_4(1:N-1)'); % should be same as PpSteady(1,1)

%% plots and disps
% disp(['MB Prediction MSE: 1-step Approx1, 1-step Approx2, 1-step Op: ', num2str([MSE_mb1, MSE_mb2, MSE_mb3opt])])
table([MSE_mb1, MSE_mb2, MSE_mb3opt, MSE_apbm_opt_0, MSE_apbm_opt_2, MSE_apbm_opt_3, MSE_apbm_opt_4]','RowNames',{'1-step Approx1', '1-step Approx2', '1-step Optimal', 'APBM Opt0', 'APBM Opt2', 'APBM Opt3', 'NN Opt 4'},'VariableNames',{'MB Prediction MSE:'})
% disp(['Steady state position variance and respective MSE (should be same): ', num2str([PpSteady(1,1), MSE_mb3opt])])
disp('Steady state position variance and respective MSE')
table([PpSteady(1,1) MSE_mb3opt]','RowNames',{'variance','MSE'},'VariableNames',{'value'})
% t.Properties.Description = 'Comparison of stead state variance and MSE';
%% --
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
fontsize=16;

figure
subplot(3,1,1)
% plot(theta(1:2e4))
plot(theta(1:N))
title('Independently varying parameter - temeperature')
xlabel('k')
ylabel('$\theta$')
ax = gca; ax.FontSize = fontsize-2;
subplot(3,1,2)
% plot(tau(1:2e4))
plot(tau(1:N))
title('Dependently varying bias time constant')
xlabel('k')
ylabel('$\tau$')
ax = gca; ax.FontSize = fontsize-2;
subplot(3,1,3)
plot(b, 'linewidth', 0.1)
hold
plot(xf(2,:), 'linewidth', 0.1)
plot(-save_nn_output_opt_0, '-','linewidth', 0.1)
title('Bias')
xlabel('k')
ylabel('bias')
legend('true','KF estim','APBM 0 estim')
xlim([0,N])
ax = gca; ax.FontSize = fontsize-2;
%%
figure; hold on; grid on
plot(-save_nn_output_opt_0, '-','linewidth', 0.1)
plot(save_nn_output_opt_2, 'linewidth', 0.1)
plot(save_bias_opt_3, 'linewidth', 0.1)
plot(b, 'linewidth', 0.1)

xlabel('k')
ylabel('bias')
legend('APBM opt 0', 'APBM opt 2', 'APBM 3', 'true')
ax = gca; ax.FontSize = fontsize-2;

figure;
plot(save_nn_params_opt_0), grid
xlabel('k', 'fontsize', fontsize) 
ylabel('\boldmath$\phi$', 'fontsize', fontsize)
ax = gca; ax.FontSize = fontsize-2;


%%
%  TODO:
%   Student-t 3 degrees of freedom added to y, theta_measured x u_measured. 
%   [Done] Modify Opt 4   
%   [Done] Implement Opt 3
%   Add the NN case for the ctr example. 
%
% True model:
% F = 1
% x = F*x + u_measured + w;
% b = f(b) + n
% y = x
% u_measured = u + bias + xi;
%
% opt 0:
% nnparams = nnparams + q
% p = F*p + u_measured - NN(u_measured, theta; nnparams)+ w;
% 
% opt 1:
% nnparams = nnparams + q
% x = F*x + [u_measured;0] - [NN(x(2), u_measured, theta; nnparams); 0]+ w;
%
% opt 2:
% nnparams = nnparams + q
% b = NN(b, u_measured, theta; nnparams) 
% p = F*p + u_measured - b + w;
%
% opt 3:
% what is x? x = [b,p] or x=[p] ?
% x = NN(F*x + [u_measured;0], u_measured, theta; nnparams) + w;
%
% TODO: p not x 
% opt 4
% p = NN(p, u_measured, theta) + w

% OPT 0
function [xa] = nn_transition_function_opt_0(x_prev, u, theta, f, nn_mlp_opt_0)
    % x_prev = [theta_prev; s_prev]
%     global nn_mlp_opt_0
%     F = 1;
    nnparams = x_prev(1:nn_mlp_opt_0.nparams);
    p = x_prev(nn_mlp_opt_0.nparams + 1: end);
    nn_mlp_opt_0.set_params(nnparams)
%     p = F*p + u - nn_mlp_opt_0.forward([u; theta]);
    p = f(p,u) - nn_mlp_opt_0.forward([u; theta]);
    xa = [nnparams; p];
end


function y = nn_reg_measurement_function_opt_0(x, nn_mlp_opt_0)
%     global nn_mlp_opt_0
    nnparams = x(1:nn_mlp_opt_0.nparams);
    p = x(nn_mlp_opt_0.nparams + 1: end);
    y = p;
    y = [nnparams; y];
end


% OPT 2
function [xa] = nn_transition_function_opt_2(x_prev, u, theta, f, nn_mlp_opt_2)
    % x_prev = [phi_prev; b_prev; p_prev]
%     global nn_mlp_opt_2
    F = 1;
    phi = x_prev(1:nn_mlp_opt_2.nparams);
    temp = x_prev(nn_mlp_opt_2.nparams + 1: end);
    b_prev = temp(1);
    p_prev = temp(2); 
    nn_mlp_opt_2.set_params(phi)
    b = b_prev + nn_mlp_opt_2.forward([b_prev; u; theta]);
%     p = F*p_prev + u - b_prev;
    p = f(p_prev, u) - b_prev;
    xa = [phi; b; p];
end


function y = nn_reg_measurement_function_opt_2(x, nn_mlp_opt_2)
%     global nn_mlp_opt_2
    nnparams = x(1:nn_mlp_opt_2.nparams);
    b = x(end-1);
    p = x(end);
    y = p;
    y = [nnparams; y];
end


% OPT 3
% x = NN(F*x + [u_measured;0], u_measured, theta; nnparams) + w;
% x = [b; p],  xa = [phi_0; phi_1; phi; x]
% x = phi0 (F*x + [u_measured;0]) + phi1 (NN(x_prev, u_measured, theta ; phi(2:))

function [xa] = nn_transition_function_opt_3(x_prev, u, theta, f, nn_mlp_opt_3)
    % x_prev = [phi_prev; b_prev; p_prev]
%     global nn_mlp_opt_3
%     F = 1;
%     F = [1, 0; -1, 1];
    % getting params
    phi_0 = x_prev(1);
    phi_1 = x_prev(2);
    phi = x_prev(3: 3 + nn_mlp_opt_3.nparams -1);
    
    % getting states
    temp = x_prev(nn_mlp_opt_3.nparams + 3: end);
    b_prev = temp(1);
    p_prev = temp(2); 
    x_prev = [b_prev; p_prev];
    
    nn_mlp_opt_3.set_params(phi)
    u_vec = [0;u];
%     x = phi_0 * (F*x_prev + u_vec) + phi_1 * nn_mlp_opt_3.forward([x_prev; u; theta]);
    x = phi_0 * f(x_prev, u_vec) + phi_1 * nn_mlp_opt_3.forward([x_prev; u; theta]);
    
    xa = [phi_0; phi_1; phi; x];
end


function y = nn_reg_measurement_function_opt_3(x, nn_mlp_opt_3)
%     global nn_mlp_opt_3
    phi_1 = x(1);
    phi_2 = x(2);
    phi = x(3: 2 + nn_mlp_opt_3.nparams);
    b = x(end-1);
    p = x(end);
    y = p;
    y = [phi_1; phi_2; phi; y];
end


% OPT 4
function [x] = nn_transition_function_opt_4(x_prev, u, theta, nn_mlp_opt_4)
    % p = NN(p, u_measured, theta) + w
    % x_prev = [phi_prev; b_prev; p_prev]
%     global nn_mlp_opt_4
    phi = x_prev(1:nn_mlp_opt_4.nparams);
    p = x_prev(end);
    nn_mlp_opt_4.set_params(phi)
    x = [phi; nn_mlp_opt_4.forward([p; u; theta])];
end

function y = nn_reg_measurement_function_opt_4(x, nn_mlp_opt_4)
%     global nn_mlp_opt_4
    nnparams = x(1:nn_mlp_opt_4.nparams);
%     b = x(end-1);
    p = x(end);
    y = p;
%     y = [nnparams; y];
    y = y;
end
