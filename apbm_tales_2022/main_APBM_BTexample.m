    % Data generator and predictors (model based, AI data-based, APBM)
% - 1D position model - random walk model with measured position increment
% - for APBM concept illustration 
% - joint paper "AI-based Modelling for Navigation and Tracking"

% ballistic target reentering the atmosphere
% h -> height
% v -> speed
% beta -> ballistic coefficient
% rho(h) -> air density rho(h) = gamma*exp(-nu*h)
% gamma = 1.754
% nu = 1.49e-4
% g -> acceleration due to gravity
% g = 9.81

%  continuous dynamic model
% hdot = -v
% vdot = - (rho(v)*g*v^2)/(2*beta) + g
% betadot = 0

% discretized dynamic model
% x = [h v beta]
% x_{k+1} = Phi x_k - G[D(x_k)-g] + w_k
% Phi = [1 -tau 0;
%        0 1    0;
%        0 0    1]
% G = [0 tau 0]'
% D(x_k) = (g*rho(x_k(1))*x_k(2)^2)/(2*x_k(3))
% Q = [q1*tau^3/2 q1*tau^2/2 0;
%      q1*tau^2/2 q1*tau     0;
%      0          0          q2*tau]
% x_o = [61000 3048 19161]
% tau = 0.1

% simplified discretized dynamic model, where the ballistic coefficient is an input 
% x = [h v]
% beta = 19161
% x_{k+1} = Phi x_k - G[D(x_k)-g] + v_k
% Phi = [1 -tau;
%        0 1   ]
% G = [0 tau]'
% D(x_k) = (g*rho(x_k(1))*x_k(2)^2)/(2*beta)
% Q = [q1*tau^3/2 q1*tau^2/2;
%      q1*tau^2/2 q1*tau    ]
% x_o = [61000 3048]
% tau = 0.1





clear; close;


% Initialisation
% - number of simulated data
N = 339; % close to surface
% discretization parameter
tau = 0.1;
% true state dim
nx = 2;
% dynamics matrices
Phi = [1 -tau;0 1];
G = [0 tau]';
% air density
gamma = 1.754;
nu = 1.49e-4;
rho = @(h) gamma*exp(-nu*h);
% acceleration due to gravity
g = 9.81;
% dynamics function D
D = @(x,beta) (g*rho(x(1))*x(2)^2)/(2*beta);
% dynamics function f
f = @(x,beta) Phi * x - G*(D(x,beta)-g);
% - initial position PDF
xinit_mean = [61000 3048]';
xinit_var = 10*eye(nx); % TBD: 
% - noise properties
% -- position white noise variance
q1 = 5;
Q = q1*[tau^3/3 tau^2/2; tau^2/2 tau];
Sq = chol(Q)';
% -- measurement noise variance
R = 0.1^2;
% -- ballistic coefficient
betaNominal = 19161;
betaFunct = @(k) betaNominal * (1+0.1*sin(k*5*1e-2)); % oscilate by 1% around nominal value % TBD: frequency can be selected
% - allocation
% -- true position, position increment, measured increment, bias, bias time constant, independent variable (e.g., temperature)
x = zeros(nx,N);
y = zeros(1,N); % noisy position measurement
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
for k = 1:N
  if k == 1
    x(:,1) =  xinit_mean + chol(xinit_var)'*randn(nx,1);
  else
    x(:,k) = f(x(:,k-1),betaFunct(k-1)) + Sq*randn(nx,1); 
  end
  y(k) = x(1,k) + sqrt(R)*trnd(3);
end

% Prediction based solely on model with nominal ballistic coefficient
xp = zeros(nx,N);
for k = 1:N
  if k == 1
    xp(:,1) =  xinit_mean + chol(xinit_var)'*randn(nx,1);
  else    
    xp(:,k) = f(xp(:,k-1),betaNominal) + Sq*randn(nx,1); 
  end 
  drag(k) = D(xp(:,k),betaNominal);
end


y_dim =1;
x_dim = 2;

% ====== CKF =======
Q_ckf = Q; %1e-2 * eye(x_dim);
P_ckf = 1e-2*eye(x_dim);
R_ckf = max(1e-6, R);

ckf_tfunc = @ckf_transition_function;         % ckf transition function
ckf_hfun = @ckf_measurement_function;         % measurement function

x0 = xinit_mean;

ckf = trackingCKF(ckf_tfunc, ckf_hfun, x0, 'ProcessNoise', Q_ckf, 'MeasurementNoise', R_ckf, 'StateCovariance', P_ckf);
save_ckf_x = zeros(N,x_dim);


% ====== APBM =======
% xa = [w_0; w_1; nnparam; h; v] = [w_0; w_1; phi; x]
% y = h
% \tilde(y) = [w_0; w_1; phi;y]
%
% x = f(x, NN_tilde(x, betaNominal; w_0. w_1, phi);
%   = phi0 * (F*x) + phi1 * (NN(x, u_measured, theta; phi(2:))


% nn_output_dim_opt_3 = x_dim;
apbm_nn_output_dim = 1;
% nn_input_dim_opt_3 = x_dim + u_dim + theta_dim;
apbm_nn_input_dim = 2 + 1 + 1;
% x_dim_opt_3 = x_dim;
x_dim_apbm = x_dim;
hidden_layers = [5];

% global nn_mlp_opt_3;              % global Neural Network needed inside the 
% APBM initialization 
apbm_nn_mlp = tmlp(apbm_nn_input_dim, apbm_nn_output_dim, hidden_layers);     % creating NN object
nnparams = apbm_nn_mlp.get_params();                        % getting NN parameters

% x0 = zeros(x_dim_apbm, 1);
x0 = xinit_mean;
% total model params = nnparams + 2, where +2 comes from phi_0 and phi_1
xa = [1e-1*randn + 1; 1e-1*randn; nnparams; x0];                                  % initial NN_CKF states

% NN process noise
Q_nn = 1e-5*eye(length(xa));
Q_nn(end-x_dim_apbm+1:end, end-x_dim_apbm+1:end) = Q; %1e-2 * eye(x_dim_apbm);

% Initial NN state cov 
P_nn = 1e-2*eye(length(xa));
P_nn(end-x_dim_apbm+1:end, end-x_dim_apbm+1:end) = 1e-2*eye(x_dim_apbm);

% noise covariance matrix for augmented likelihood model (for
% regularization)
lambda = 1e-3;
R_nn = (1/lambda)*eye(length(xa)-1);
% TODO: set a better Ry
R_nn(end - 1, end - 1) = 1e-6;
R_nn(end-y_dim+1:end,end-y_dim+1:end) = max(1e-6, R);


apbm_nn_tfunc = @apbm_transition_function;             % APBM transition function
apbm_nn_hfun = @apbm_reg_measurement_function;         % regularized measurement function

apbm_ckf = trackingCKF(apbm_nn_tfunc, apbm_nn_hfun, xa, 'ProcessNoise', Q_nn, 'MeasurementNoise', R_nn, 'StateCovariance', P_nn);

save_apbm_ckf_x = zeros(N,x_dim);
save_apbm_nn_params = zeros(N, apbm_nn_mlp.nparams + 2);
% zero vector for likelihood augmentation
zero_meas = zeros(apbm_nn_mlp.nparams,1);
% adding regularization for phi_0=1 and phi_1=0
apbm_aug_measurements = [1; 0; zero_meas];

for k = 1:N
    
    % -------- CKF --------
    [ckf_xPred, ~] = predict(ckf, betaNominal, f);
    
    % correct with augmented likelihood function:
    [ckf_xCorr, ~] = correct(ckf, y(k));     
    
    save_ckf_x(k,:) = ckf_xPred;
    
    
    % -------- APBM --------
    [apbm_xPred, ~] = predict(apbm_ckf, betaNominal, k, f, apbm_nn_mlp);
    
    % correct with augmented likelihood function:
    [apbm_ckf_xCorr, ~] = correct(apbm_ckf, [apbm_aug_measurements ; y(k)], apbm_nn_mlp);     
    
    % getting nn params
    save_apbm_nn_params(k,:) = apbm_ckf_xCorr(1:end-x_dim_apbm);
    % getting only states (not parameters)
   
    % saving nn_ckf estimated states
    apbm_ckf_xPred = apbm_xPred(end-x_dim+1 : end);
    save_apbm_ckf_x(k,:) = apbm_ckf_xPred;
end



%%

% Plotting
figure
t = [1:N];
subplot(2,2,1)
plot(t,x(2,:),'-',t,xp(2,:),'--',t, save_apbm_ckf_x(:,2),':', t, save_ckf_x(:,2))
legend('true','model betaNominal', 'APBM', 'CKF')
title('velocity')
subplot(2,2,2)
plot(t,x(1,:),'-',t,xp(1,:),'--', t, save_apbm_ckf_x(:,1),':', t, save_ckf_x(:,1))
legend('true','model betaNominal', 'APBM', 'CKF')
title('altitude')
subplot(2,2,3)
plot(t,x(2,:)-xp(2,:),'b', t, x(2,:)-save_apbm_ckf_x(:,2)', t, x(2,:)-save_ckf_x(:,2)')
legend('nominal model', 'APBM', 'CKF')
title('velocity difference')
subplot(2,2,4)
plot(t,x(1,:)-xp(1,:),'b', t, x(1,:)-save_apbm_ckf_x(:,1)', t, x(1,:)-save_ckf_x(:,1)')
legend('Nominal Model', 'APBM', 'CKF')
title('altitude difference')
figure
subplot(1,2,1)
plot(t,betaFunct(t))
title('ballistic coeff.')
subplot(1,2,2)
plot(t,drag)
title('drag')
return




% xa = [phi;x] -> augmented states

% phi = phi
% x = f_APBM(x)
% f_APBM(x) =  f(x,beta_NN(k))
% [phi, y] = h(x, phi)
% beta_NN(k) = w0 * betaNominal + w1*NN(x, betaNominal, k).

function [xa] = apbm_transition_function(x_prev, betaNominal, k, f, apbm_nn_mlp)
    % getting params
    w_0 = x_prev(1);
    w_1 = x_prev(2);
    phi = x_prev(3: 3 + apbm_nn_mlp.nparams -1);
    
    % getting states
    temp = x_prev(apbm_nn_mlp.nparams + 3: end);
    h_prev = temp(1);
    v_prev = temp(2); 
    x_prev = [h_prev; v_prev];
    
    apbm_nn_mlp.set_params(phi)
%     x = w_0 * f(x_prev, betaNominal) + w_1 * apbm_nn_mlp.forward([x_prev; k; betaNominal]);
    beta =  w_0 * betaNominal + w_1 * apbm_nn_mlp.forward([x_prev; k; betaNominal]);
    x = f(x_prev, beta);
    
    xa = [w_0; w_1; phi; x];
end


function y = apbm_reg_measurement_function(x, nn_mlp_opt_3)
%     global nn_mlp_opt_3
    w_1 = x(1);
    w_2 = x(2);
    phi = x(3: 2 + nn_mlp_opt_3.nparams);
    h = x(end-1);
    v = x(end);
    y = h;
    y = [w_1; w_2; phi; y];
end



function [x] = ckf_transition_function(x_prev, betaNominal, f)    
    x = f(x_prev, betaNominal);
end

function y = ckf_measurement_function(x)
    h = x(1);
    v = x(2);
    y = h;
end



%% TODO 
% Fit the different styles in section III. 

% APBM f_apbm(x,betaNominal) = f(x,betaNominal) + NN(x, betaNominal)
% NN  f_NN(x,betaNominal?) = NN(x, betaNominal?)
% Compute RMSE (maybe use log scale)
% CDF of error
% Change figure 1 to be more general: data-driven model

