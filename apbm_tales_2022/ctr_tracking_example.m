% Implements the APBM proposed in
% Imbiriba et. al., Hybrid Neural Network Augmented Physics-based Models
% for Nonlinear Filtering.
%
% Author: Tales Imbiriba.

clear; 
close all;
% rng(1);

% global nn_mlp;              % global Neural Network needed inside the 
                            % the transition and measurement functions
                            % defined in the end of this file.

Nt = 2000;                  % number of iterations
Ts = 1;                   % time-step

x_dim = 4;                  % number of states
y_dim = 2;                  % number of measurements
x0 = [10,0,10,0]';          % initial state for data generation
P0 = diag([0.1, 0.01, 0.1, 0.01]);  % initial state covariance matrix
                            % generation loop

% Generating elements of the process covariance matrix Qp = Gamma*Q*Gamma'
q = sqrt(0.1);
Q = q^2 * eye(2);

Gamma = [Ts^2/2 0;
        Ts   0;
        0  Ts^2/2;
        0  Ts];
    
r = sqrt(1e-2);         % noise covariance
R = r^2 * eye(2);       % noise covariance matrix


save_x_cell = {Nr};        % true states
save_apbm_cell = {};     % apbm + ckf
save_cv_cell = {};       % cv + ckf
save_nn_cell = {};       % nn + ckf
save_tm_cell = {};       % true model + ckf


for r=1:Nruns 
    x = x0; P = P0;             % initializing variables used in the data 

    % Omega init and covariance
    Omega = 0.05*pi;
    Q_Omega = 1e-4;

    % defining transition and measurement functions

    tfunc = @ctr_transition_function;               % data_gen cos turning rate 
                                                    % transition function                                                
    cvtfunc = @const_vel_transition_function;       % const vel trans. function

    % data gen measurement function
    hfun = @(x) [30 - 10*log10(norm(-x(1:2:3))^2.2); atan2(x(3),x(1))];

    % ckf = trackingCKF(tfunc, hfun, x, 'ProcessNoise', Gamma*Q*Gamma', 'MeasurementNoise', R);
    ckf = trackingCKF(cvtfunc, hfun, x, 'ProcessNoise', Gamma*Q*Gamma', 'MeasurementNoise', R);

    % ============== APBM ==============

    % neural net measurement function
    % nn_hfun = @nn_measurement_function;
    apbm_hfun = @apbm_reg_measurement_function;
    apbm_tfunc = @apbm_transition_function;             % APBM transition function

    % APBM initialization 
    apbm_nn_mlp = tmlp(length(x0), length(x0), [2]);         % creating NN object
    theta = apbm_nn_mlp.get_params();                        % getting NN parameters
    w0 = [1;0] + 1e-2*randn(2,1);
    x_nn = [theta; w0; x0];                                  % initial NN_CKF states

    % NN process noise
    % Q_nn = q^2*eye(length(x_nn));                       
    Q_nn = 1e-5*eye(length(x_nn));
    Q_nn(end-x_dim+1:end, end-x_dim+1:end) = Gamma*Q*Gamma';

    % Initial NN state cov 
    P_apbm = 1e-2*eye(length(x_nn));
    P_apbm(end-x_dim+1:end, end-x_dim+1:end) = 1e-2*eye(x_dim);

    % noise covariance matrix for augmented likelihood model (for
    % regularization)
    lambda = 0.01;
    R_apbm = (1/lambda)*eye(length(x_nn)-2);
    R_apbm(end-y_dim+1:end,end-y_dim+1:end) = R;

    % create CKF filter
    apbm_ckf = trackingCKF(apbm_tfunc, apbm_hfun, x_nn, 'ProcessNoise', Q_nn, 'MeasurementNoise', R_apbm, 'StateCovariance', P_apbm);


    % ============== CKF True Model ==============

    % neural net measurement function
    % nn_hfun = @nn_measurement_function;
    % ckft_hfun = @hfun;
    cktf_tfunc = @ctr_transition_function;       
    x0_tm = [x;Omega];
    Q_tm = [Gamma*Q*Gamma', zeros(4,1); zeros(1,4), Q_Omega];

    % ckf true model
    ckf_tm = trackingCKF(cktf_tfunc, hfun, x0_tm, 'ProcessNoise', Q_tm, 'MeasurementNoise', R);

    % ============== NN ==============
    % nn_hfun = @apbm_reg_measurement_function;             % NN measurement function
    % nn_hfun = @nn_measurement_function;
    nn_hfun = @nn_reg_measurement_function;
    nn_tfunc = @nn_transition_function;             % NN transition function

    nn_xdim = 2;
    % NN initialization 
    nn_mlp = tmlp(length(x0)-2, length(x0)-2, [4]);         % creating NN object
    theta = nn_mlp.get_params();                        % getting NN parameters
    x_nn = [theta;x0(1);x0(3)];                                  % initial NN_CKF states

    % NN process noise
    Q_nn = 1e-5*eye(length(x_nn));
    % Q_nn(end-x_dim+1:end, end-x_dim+1:end) = Gamma*Q*Gamma';

    % Initial NN state cov 
    P_nn = 1e-2*eye(length(x_nn));
    % P_nn(end-x_dim+1:end, end-x_dim+1:end) = 1e-2*eye(x_dim);

    % noise covariance matrix for augmented likelihood model (for
    % regularization)
    lambda = 100;
    R_nn = (1/lambda)*eye(length(x_nn));
    R_nn(end-y_dim+1:end,end-y_dim+1:end) = R;

    % create CKF filter
    nn_ckf = trackingCKF(nn_tfunc, nn_hfun, x_nn, 'ProcessNoise', Q_nn, 'MeasurementNoise', R_nn, 'StateCovariance', P_nn);

    % save variables
    save_x = zeros(Nt,2);
    save_Omega = zeros(Nt,1);
    save_y = zeros(Nt,2);
    save_ckf_tm_x = zeros(Nt,2);
    save_ckf_tm_Omega = zeros(Nt,1);
    save_tm_ckf_x_mmse = zeros(Nt,2);
    save_apbm_ckf_x_mmse = zeros(Nt,2);
    save_nn_ckf_x_mmse = zeros(Nt,2);
    save_ckf_x_mmse = zeros(Nt,2);
    save_apbm_params = zeros(Nt, apbm_nn_mlp.nparams + 2);
    save_nn_params = zeros(Nt, nn_mlp.nparams);

    % zero vector for likelihood augmentation
    zero_meas = zeros(apbm_nn_mlp.nparams,1);



    for n=1:Nt
        % data generation
        x = tfunc([x;Omega], Ts) + [Gamma * mvnrnd([0, 0], Q)'; sqrt(Q_Omega)*randn];
        y = hfun(x) + mvnrnd([0, 0], R)';

        Omega = x(end);
        x = x(1:end-1);

        % standard CKF (constant velocity)
        [ckf_xPred, ckf_pPred] = predict(ckf, Ts);
        [ckf_xCorr, ckf_pCorr] = correct(ckf, y);

        % ckf with true model (tm)
        [ckf_tm_xPred, ckf_tm_pPred] = predict(ckf_tm, Ts);
        [ckf_tm_xCorr, ckf_tm_pCorr] = correct(ckf_tm, y);    

        % APBM CKF
        if n>1
            P_old = apbm_pPred;
        end
        [apbm_xPred, apbm_pPred] = predict(apbm_ckf, Ts, apbm_nn_mlp);
        % correct with augmented likelihood function:
        [apbm_ckf_xCorr, apbm_ckf_pCorr] = correct(apbm_ckf, [zero_meas; 1; 0; y], apbm_nn_mlp);     

        % NN CKF
        [nn_xPred, nn_pPred] = predict(nn_ckf, Ts, nn_mlp);
        % correct with augmented likelihood function:
        [nn_ckf_xCorr, nn_ckf_pCorr] = correct(nn_ckf, [zero_meas ;y], nn_mlp);     

        % testing/ making things flowing
        P = apbm_ckf.StateCovariance;
        if max(eig(P))> 1e4
            disp(['max(eig(P))> 1e4 -> ', num2str(max(eig(P)))])
            apbm_ckf.StateCovariance = P_apbm;
        end
        min_eig = min(eig(P));
    %     if max(eig(P)) > 1000
    %         disp('fudeu')
    %         eig(P)
    %         break;
    %         apbm_ckf.StateCovariance = 0.1*P_nn;
    %     end
        if min_eig < 1e-8
            disp('here 1')
            apbm_ckf.StateCovariance = apbm_ckf.StateCovariance  + 1*min_eig*eye(size(apbm_ckf.StateCovariance));
        end
        P_old = P;
        % testing/ making things flowing
        P = nn_ckf.StateCovariance;
        min_eig = min(eig(P));
        if min_eig < 1e-4
    %         disp('here')
            nn_ckf.StateCovariance = nn_ckf.StateCovariance  + 1*min_eig*eye(size(nn_ckf.StateCovariance));
        end


        % saving true states for plotting and error computations
        save_x(n,:) = [x(1),  x(3)]';
        save_Omega(n,:) = Omega;
        save_y(n,:) = y';

        % getting apbm nn params
        save_apbm_params(n,:) = apbm_ckf_xCorr(1:end-x_dim);
        % getting only states (not parameters)
        apbm_ckf_xCorr = apbm_ckf_xCorr(end-x_dim+1:end);

        % getting nn params
        save_nn_params(n,:) = nn_ckf_xCorr(1:end-nn_xdim);
        % getting only states (not parameters)
        nn_ckf_xCorr = nn_ckf_xCorr(end-nn_xdim+1:end);

        % saving ckf and nn_ckf estimated states
        save_ckf_x_mmse(n,:) = [ckf_xCorr(1),  ckf_xCorr(3)]';
        save_tm_ckf_x_mmse(n,:) = [ckf_tm_xCorr(1),  ckf_tm_xCorr(3)]';
        save_apbm_ckf_x_mmse(n,:) = [apbm_ckf_xCorr(1),  apbm_ckf_xCorr(3)]';
        save_nn_ckf_x_mmse(n,:) = [nn_ckf_xCorr(1),  nn_ckf_xCorr(2)]';
        
        
    end
    save_x_cell{r} = save_x;        
    save_apbm_cell{r} =  save_apbm_ckf_x_mmse;     
    save_cv_cell{r} = save_ckf_x_mmse;    
    save_nn_cell{r} = save_nn_ckf_x_mmse;    
    save_tm_cell{r} = save_tm_ckf_x_mmse;
end
 
%% Plots
fontsize=16;
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
figure;
plot(save_x(:,1),save_x(:,2),'-*','LineWidth',.1), hold on, grid
% plot(save_y(:,1),save_y(:,2),'.','LineWidth',1)
plot(save_apbm_ckf_x_mmse(:,1),save_apbm_ckf_x_mmse(:,2), '-s','LineWidth',.1)
plot(save_nn_ckf_x_mmse(:,1),save_nn_ckf_x_mmse(:,2), '-s','LineWidth',.1)
plot(save_ckf_x_mmse(:,1),save_ckf_x_mmse(:,2), '-^','LineWidth',.1)
plot(save_tm_ckf_x_mmse(:,1),save_tm_ckf_x_mmse(:,2), '-o','LineWidth',.1)

% scatter(save_y(:,1), save_y(:,2))
scatter(0,0,'xk', 'linewidth',2)
ax = gca; ax.FontSize = fontsize-2;
xlabel('x [m]','fontsize', fontsize) 
ylabel('y [m]', 'fontsize', fontsize)
legend('True', 'APBM', 'NN','CV-CKF','TM-CKF','Sensor','Location','northwest', 'fontsize', fontsize-2)
% rectangle('Position',[-1 -1 2 2],'EdgeColor','k'), daspect([1 1 1])
% text(-2,2,'Sensor', 'fontsize', fontsize)

RMSE_APBM_CKF_MMSE = sqrt((norm(save_apbm_ckf_x_mmse - save_x).^2)/length(save_x))
RMSE_NN_CKF_MMSE = sqrt((norm(save_nn_ckf_x_mmse - save_x).^2)/length(save_x))
RMSE_CKF_MMSE = sqrt((norm(save_ckf_x_mmse - save_x).^2)/length(save_x))
RMSE_CKF_TM_MMSE = sqrt((norm(save_tm_ckf_x_mmse - save_x).^2)/length(save_x))

tvec = [0:Nt-1]*Ts;
figure;
plot(tvec,sum(sqrt((save_apbm_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',.1), hold on, grid
plot(tvec,sum(sqrt((save_nn_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',.1)
plot(tvec,sum(sqrt((save_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',.1)
xlabel('time [s]', 'fontsize', fontsize), 
ylabel('RMSE [m]', 'fontsize', fontsize)
legend('APBM', 'NN', 'CV', 'fontsize', fontsize-2)
ax = gca; ax.FontSize = fontsize-2;

figure;
plot(tvec, save_apbm_params), grid
xlabel('time [s]', 'fontsize', fontsize) 
ylabel('\boldmath$\theta$', 'fontsize', fontsize)
ax = gca; ax.FontSize = fontsize-2;
%% Functions

function [x] = const_vel_transition_function(x_prev, Ts)
    F = [1 Ts 0 0;
         0 1 0 0;
         0 0 1 Ts;
         0 0 0 1];
     x = F*x_prev;
end

function [x] = ctr_transition_function(x_prev, Ts)
%     Omega_prev = 0.05*pi;
    Omega_prev = x_prev(end);
    s_prev = x_prev(1:end-1);
    OTs = Omega_prev*Ts;
    if Omega_prev ==0
        B = eye(4);
    else
        B = [1, sin(OTs)/Omega_prev, 0, -(1-cos(OTs))/Omega_prev;
             0, cos(OTs), 0, -sin(OTs);
             0, (1-cos(OTs))/Omega_prev, 1, sin(OTs)/Omega_prev;
             0, sin(OTs), 0, cos(OTs)];
    end
    s = B*s_prev;
    Omega = Omega_prev;
    x = [s; Omega];
%     Omega = Omega_prev;
%     x = [s; Omega]; 
end

function [x] = nn_transition_function(x_prev, Ts, nn_mlp)
    % x_prev = [theta_prev; s_prev]
%     global nn_mlp
    theta = x_prev(1:nn_mlp.nparams);
    s = x_prev(nn_mlp.nparams + 1: end);
    nn_mlp.set_params(theta)
    s = Ts * nn_mlp.forward(s);
    x = [theta; s];
end

function [x] = apbm_transition_function(x_prev, Ts, nn_mlp)
    % x_prev = [theta_prev, w_prev; s_prev]
%     global nn_mlp
    F = [1 Ts 0 0;
     0 1 0 0;
     0 0 1 Ts;
     0 0 0 1];
    theta = x_prev(1:nn_mlp.nparams);
    w = x_prev(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
    s = x_prev(nn_mlp.nparams + 3: end);
    nn_mlp.set_params(theta)
    s = w(1)*F*s + w(2)*nn_mlp.forward(s);
    x = [theta; w; s];
end

function y = nn_measurement_function(x, nn_mlp)
%     global nn_mlp
%     theta = x_prev(1:nn_mlp.nparams);
    s = x(nn_mlp.nparams + 1: end);
%     y = [30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
    y = [30 - 10*log10(norm(-s)^2.2); atan2(s(2),s(1))];
end

function y = nn_reg_measurement_function(x, nn_mlp)
%     global nn_mlp
    theta = x(1:nn_mlp.nparams);
    s = x(nn_mlp.nparams + 1: end);
%     y = [30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
    y = [theta; 30 - 10*log10(norm(-s)^2.2); atan2(s(2),s(1))];
end


function y = apbm_reg_measurement_function(x, nn_mlp)
%     global nn_mlp
    theta = x(1:nn_mlp.nparams);
    w = x(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
    s = x(nn_mlp.nparams + 3: end);
    y = [30 - 10*log10(norm(-s(1:2:3))^2.2); atan2(s(3),s(1))];
    y = [theta; w; y];
end



%% TODO
% Implement CKF with the true model.



