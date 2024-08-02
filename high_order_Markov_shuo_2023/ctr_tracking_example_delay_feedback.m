% Implements the APBM proposed in
% Imbiriba et. al., Hybrid Neural Network Augmented Physics-based Models
% for Nonlinear Filtering.
%
% Author: Tales Imbiriba.

clear; 
close all;
rng(3);

%% PARAMETER SETTINGS
ref = load("refState.mat");
% global nn_mlp;              % global Neural Network needed inside the 
                            % the transition and measurement functions
                            % defined in the end of this file.

Nruns = 50;
Nt = 500;                  % number of iterations
Ts = 1;                   % time-step

x_dim = 4;                  % number of states
y_dim = 2;                  % number of measurements
x0 = [10,0,10,0]';          % initial state for data generation
P0 = diag([0.1, 0.01, 0.1, 0.01]);  % initial state covariance matrix
                            % generation loop

% Generating elements of the process covariance matrix Qp = Gamma*Q*Gamma'
% q = sqrt(0.1);
q = sqrt(0.01);
Q = q^2 * eye(2);

Gamma = [Ts^2/2 0;
        Ts   0;
        0  Ts^2/2;
        0  Ts];

% Q_Omega = 1e-4;
Q_Omega = 1e-4;
P_Omega_0 = 1e-2;
    
r = sqrt(1e-2);         % noise covariance
R = r^2 * eye(2);       % noise covariance matrix
% R = diag([1,0.1]);


lambda_apbm = 0.05;
lambda_nn = 0.05;

save_x_cell = {Nruns, 1};        % true states
% save_apbm_cell = {};     % apbm + ckf
% save_cv_cell = {};       % cv + ckf
% save_nn_cell = {};       % nn + ckf
% save_tm_cell = {};       % true model + ckf

save_apbm_rmse_mlp = zeros(Nruns,1);
save_apbm_rmse_mlp_nonMk = zeros(Nruns,1);
save_apbm_rmse_rnn = zeros(Nruns,1);
save_cv_rmse  = zeros(Nruns,1);
save_nn_rmse = zeros(Nruns,1);
save_tm_rmse = zeros(Nruns,1);

for r=1:Nruns
    r
    %% MODEL SETTINGS
    x = x0 ; P = P0;             % initializing variables used in the data 
    x_init = mvnrnd(x0, P0)';

    % Omega init and covariance
    Omega = 0.05*pi;
    Omega_init = mvnrnd(Omega, P_Omega_0)';
    
    % defining transition and measurement functions

    tfunc = @cv_delay_feedback_function;            % data_gen cos turning rate 
                                                    % transition function                                                
    cvtfunc = @const_vel_transition_function;       % const vel trans. function

    % data gen measurement function
    hfun = @(x) [30 - 10*log10(norm(-x(1:2:3))^2.2); atan2(x(3),x(1))];

    % ckf = trackingCKF(tfunc, hfun, x, 'ProcessNoise', Gamma*Q*Gamma', 'MeasurementNoise', R);
    ckf = trackingCKF(cvtfunc, hfun, x_init, 'ProcessNoise', Gamma*Q*Gamma', 'MeasurementNoise', R, 'StateCovariance', P);

     % ============== APBM MLP ==============

    % neural net measurement function
    % nn_hfun = @nn_measurement_function;
    apbm_hfun = @apbm_reg_measurement_function;
    apbm_tfunc_mlp = @apbm_transition_function;             % APBM transition function

    % APBM initialization 
    apbm_nn_mlp = tmlp(length(x0), length(x0), [5]);         % creating NN object
    theta = apbm_nn_mlp.get_params();                        % getting NN parameters
    w0 = [1;0] + 1e-2*randn(2,1);
    x_nn_mlp = [theta; w0; x_init];                                  % initial NN_CKF states

    % NN process noise
    % Q_nn = q^2*eye(length(x_nn));                       
    Q_nn_mlp = 1e-6*eye(length(x_nn_mlp));
    Q_nn_mlp(end-x_dim+1:end, end-x_dim+1:end) = Gamma*Q*Gamma';

    % Initial NN state cov 
    P_apbm_mlp = 1e-2*eye(length(x_nn_mlp));
    P_apbm_mlp(end-x_dim+1:end, end-x_dim+1:end) = 1e-2*eye(x_dim);

    % noise covariance matrix for augmented likelihood model (for
    % regularization)
    R_apbm_mlp = (1/lambda_apbm)*eye(length(x_nn_mlp)-2);
    R_apbm_mlp(end-y_dim+1:end,end-y_dim+1:end) = R;

    % create CKF filter
    apbm_ckf_mlp = trackingCKF(apbm_tfunc_mlp, apbm_hfun, x_nn_mlp, 'ProcessNoise', Q_nn_mlp, 'MeasurementNoise', R_apbm_mlp, 'StateCovariance', P_apbm_mlp);

    % ============== APBM MLP non-Markovian ==============

    % neural net measurement function
    % nn_hfun = @nn_measurement_function;
    apbm_hfun = @apbm_reg_measurement_function;
    apbm_tfunc_nonMk = @apbm_transition_function_nonMk;             % APBM transition function

    % APBM initialization 
    apbm_nn_mlp_nonMk = tmlp(length(x0)*5, length(x0), [5]);         % creating NN object
    theta_nonMk = apbm_nn_mlp_nonMk.get_params();                        % getting NN parameters
    w0_nonMk = [1;0] + 1e-2*randn(2,1);
    x_nn_mlp_nonMk = [theta_nonMk; w0_nonMk; x_init];                                  % initial NN_CKF states

    % NN process noise
    % Q_nn = q^2*eye(length(x_nn));                       
    Q_nn_nonMk = 1e-6*eye(length(x_nn_mlp_nonMk));
    Q_nn_nonMk(end-x_dim+1:end, end-x_dim+1:end) = Gamma*Q*Gamma';

    % Initial NN state cov 
    P_apbm_nonMk = 1e-2*eye(length(x_nn_mlp_nonMk));
    P_apbm_nonMk(end-x_dim+1:end, end-x_dim+1:end) = 1e-2*eye(x_dim);

    % noise covariance matrix for augmented likelihood model (for
    % regularization)
    R_apbm_nonMk = (1/lambda_apbm)*eye(length(x_nn_mlp_nonMk)-2);
    R_apbm_nonMk(end-y_dim+1:end,end-y_dim+1:end) = R;

    % create CKF filter
    apbm_ckf_nonMk = trackingCKF(apbm_tfunc_nonMk, apbm_hfun, x_nn_mlp_nonMk, 'ProcessNoise', Q_nn_nonMk, 'MeasurementNoise', R_apbm_nonMk, 'StateCovariance', P_apbm_nonMk);

    % ============== APBM RNN ==============

    % neural net measurement function
    % nn_hfun = @nn_measurement_function;
    apbm_hfun = @apbm_reg_measurement_function;
    apbm_tfunc_rnn = @apbm_transition_function_rnn;             % APBM transition function

    % APBM initialization 
    apbm_nn_rnn = RNN([length(x0),5], length(x0), 5);         % creating NN object
    theta_rnn = apbm_nn_rnn.get_params();                        % getting NN parameters
    w0_rnn = [1;0] + 1e-2*randn(2,1);
    x_rnn = [theta_rnn; w0_rnn; x_init];                                  % initial NN_CKF states

    % NN process noise
    % Q_nn = q^2*eye(length(x_nn));                       
    Q_nn_rnn = 1e-6*eye(length(x_rnn));
    Q_nn_rnn(end-x_dim+1:end, end-x_dim+1:end) = Gamma*Q*Gamma';

    % Initial NN state cov 
    P_apbm_rnn = 1e-2*eye(length(x_rnn));
    P_apbm_rnn(end-x_dim+1:end, end-x_dim+1:end) = 1e-2*eye(x_dim);

    % noise covariance matrix for augmented likelihood model (for
    % regularization)
    R_apbm_rnn = (1/lambda_apbm)*eye(length(x_rnn)-2);
    R_apbm_rnn(end-y_dim+1:end,end-y_dim+1:end) = R;

    % create CKF filter
    apbm_ckf_nonMk_rnn = trackingCKF(apbm_tfunc_rnn, apbm_hfun, x_rnn, 'ProcessNoise', Q_nn_rnn, 'MeasurementNoise', R_apbm_rnn, 'StateCovariance', P_apbm_rnn);

    % ============== CKF True Model ==============

    % neural net measurement function
    % nn_hfun = @nn_measurement_function;
    % ckft_hfun = @hfun;
    cktf_tfunc = @cv_delay_feedback_function;       
    x0_tm = [x_init; Omega_init];
    Q_tm = [Gamma*Q*Gamma', zeros(4,1); zeros(1,4), Q_Omega];
    P_0tm = 1e-2*eye(length(x0_tm));
    % ckf true model
    ckf_tm = trackingCKF(cktf_tfunc, hfun, x0_tm, 'ProcessNoise', Q_tm, 'MeasurementNoise', R,  'StateCovariance', P_0tm);

    % ============== NN ==============
    % nn_hfun = @apbm_reg_measurement_function;             % NN measurement function
    % nn_hfun = @nn_measurement_function;
    nn_hfun = @nn_reg_measurement_function;
    nn_tfunc = @nn_transition_function;             % NN transition function

    nn_xdim = 2;
    % NN initialization 
    nn_mlp = tmlp(length(x0)-2, length(x0)-2, [5]);         % creating NN object
    theta_nonMk = nn_mlp.get_params();                        % getting NN parameters
    x_nn = [theta_nonMk; x_init(1);x_init(3)];                                  % initial NN_CKF states

    % NN process noise
    Q_nn_nonMk = 1e-6*eye(length(x_nn));
    % Q_nn(end-x_dim+1:end, end-x_dim+1:end) = Gamma*Q*Gamma';

    % Initial NN state cov 
    P_nn = 1e-2*eye(length(x_nn));
    % P_nn(end-x_dim+1:end, end-x_dim+1:end) = 1e-2*eye(x_dim);

    % noise covariance matrix for augmented likelihood model (for
    % regularization)
    R_nn = (1/lambda_nn)*eye(length(x_nn));
    R_nn(end-y_dim+1:end,end-y_dim+1:end) = R;

    % create CKF filter
    nn_ckf = trackingCKF(nn_tfunc, nn_hfun, x_nn, 'ProcessNoise', Q_nn_nonMk, 'MeasurementNoise', R_nn, 'StateCovariance', P_nn);

    % save variables
    save_x = zeros(Nt,2);
    save_Omega = zeros(Nt,1);
    save_y = zeros(Nt,2);
    save_ckf_tm_x = zeros(Nt,2);
    save_ckf_tm_Omega = zeros(Nt,1);
    save_tm_ckf_x_mmse = zeros(Nt,2);
    save_apbm_ckf_x_mmse_mlp = zeros(Nt,2);
    save_apbm_ckf_x_mmse_nonMk = zeros(Nt,2);
    save_apbm_ckf_x_mmse_rnn = zeros(Nt,2);
    save_nn_ckf_x_mmse = zeros(Nt,2);
    save_ckf_x_mmse = zeros(Nt,2);
    save_apbm_params_mlp = zeros(Nt, apbm_nn_mlp.nparams + 2);
    save_apbm_params_nonMk = zeros(Nt, apbm_nn_mlp_nonMk.nparams + 2);
    save_apbm_params_rnn = zeros(Nt, apbm_nn_rnn.nparams + 2);
    save_nn_params = zeros(Nt, nn_mlp.nparams);

    % zero vector for likelihood augmentation
    theta_bar_apbm_mlp = zeros(apbm_nn_mlp.nparams,1);
    theta_bar_apbm_nonMk = zeros(apbm_nn_mlp_nonMk.nparams,1);
    theta_bar_apbm_rnn = zeros(apbm_nn_rnn.nparams,1);
    theta_bar_nn = zeros(nn_mlp.nparams,1);

    % ============== non-Markovian True Model ==============
    % define non-Markovian transition model
    cktf_tfunc_nMk = @ctr_transition_function_nMk;

    %% TIME STEP RUNNING
    for n=1:Nt
        %% Data Generation with True Model
        nonMkStep = 3;
        if n <= nonMkStep
            x = cktf_tfunc([x;Omega], Ts, zeros(5, 1), ref.refState(:, n)) + mvnrnd(zeros(1,5), Q_tm)';
            switch n
                case 1
                    x_k_3 = x;
                case 2
                    x_k_2 = x;
                case 3
                    x_k_1 = x;
                % case 4 
                %     x_k_1 = x;
            end
        else  
            x = cktf_tfunc([x;Omega], Ts, x_k_3, ref.refState(:, n)) + mvnrnd(zeros(1,5), Q_tm)'; 
            % x_k_4 = x_k_3;
            x_k_3 = x_k_2;
            x_k_2 = x_k_1;
            x_k_1 = x;
        end

        y = hfun(x) + mvnrnd([0, 0], R)';

        Omega = x(end);
        x = x(1:end-1);
        %% Normal CKF
        % standard CKF (constant velocity)
        [ckf_xPred, ckf_pPred] = predict(ckf, Ts);
        [ckf_xCorr, ckf_pCorr] = correct(ckf, y);

        % ckf with true model (tm)
        if n <= nonMkStep
            [ckf_tm_xPred, ckf_tm_pPred] = predict(ckf_tm, Ts, zeros(5, 1), ref.refState(:, n));
            [ckf_tm_xCorr, ckf_tm_pCorr] = correct(ckf_tm, y); 
            switch n
                case 1
                    x_k_3_tm = ckf_tm_xCorr;
                case 2
                    x_k_2_tm = ckf_tm_xCorr;
                case 3
                    x_k_1_tm = ckf_tm_xCorr;
            end
        else
            [ckf_tm_xPred, ckf_tm_pPred] = predict(ckf_tm, Ts, x_k_3_tm, ref.refState(:, n));
            [ckf_tm_xCorr, ckf_tm_pCorr] = correct(ckf_tm, y);
            x_k_3_tm = x_k_2_tm;
            x_k_2_tm = x_k_1_tm;
            x_k_1_tm = ckf_tm_xCorr;
        end

        %% APBM MLP
        nonMkStep_nn = 5;
        % APBM MLP Markovian
        [apbm_xPred_mlp, apbm_pPred_mlp] = predict(apbm_ckf_mlp, Ts, apbm_nn_mlp);
        % correct with augmented likelihood function:
        [apbm_ckf_xCorr_mlp, apbm_ckf_pCorr_mlp] = correct(apbm_ckf_mlp, [theta_bar_apbm_mlp; 1; 0; y], apbm_nn_mlp);  

        % APBM MLP non-Markovian
        if n <= nonMkStep_nn
            [apbm_xPred_nonMk, apbm_pPred_nonMk] = predict(apbm_ckf_nonMk, Ts, apbm_nn_mlp_nonMk, ...
                zeros(4, 1), zeros(4, 1), zeros(4, 1), zeros(4, 1), zeros(4, 1));
            [apbm_ckf_xCorr_nonMk, apbm_ckf_pCorr_nonMk] = correct(apbm_ckf_nonMk, [theta_bar_apbm_nonMk; 1; 0; y], apbm_nn_mlp_nonMk);
            switch n
                case 1
                    x_k_5_mlp = apbm_ckf_xCorr_nonMk(end-3:end);
                case 2
                    x_k_4_mlp = apbm_ckf_xCorr_nonMk(end-3:end);
                case 3
                    x_k_3_mlp = apbm_ckf_xCorr_nonMk(end-3:end);
                case 4
                    x_k_2_mlp = apbm_ckf_xCorr_nonMk(end-3:end);
                case 5
                    x_k_1_mlp = apbm_ckf_xCorr_nonMk(end-3:end);
            end
        else
            [apbm_xPred_nonMk, apbm_pPred_nonMk] = predict(apbm_ckf_nonMk, Ts, apbm_nn_mlp_nonMk, ...
                x_k_5_mlp, x_k_4_mlp, x_k_3_mlp, x_k_2_mlp, x_k_1_mlp);
            % correct with augmented likelihood function:
            [apbm_ckf_xCorr_nonMk, apbm_ckf_pCorr_nonMk] = correct(apbm_ckf_nonMk, [theta_bar_apbm_nonMk; 1; 0; y], apbm_nn_mlp_nonMk);
            % update memory
            x_k_5_mlp = x_k_4_mlp;
            x_k_4_mlp = x_k_3_mlp;
            x_k_3_mlp = x_k_2_mlp;
            x_k_2_mlp = x_k_1_mlp;
            x_k_1_mlp = apbm_ckf_xCorr_nonMk(end-3:end);
        end
                 
        %% APBM RNN
        % ABPM RNN
        if n <= nonMkStep_nn
            [apbm_xPred_rnn, apbm_pPred_rnn] = predict(apbm_ckf_nonMk_rnn, Ts, apbm_nn_rnn, ...
                zeros(4, 1), zeros(4, 1), zeros(4, 1), zeros(4, 1), zeros(4, 1));
             [apbm_ckf_xCorr_rnn, apbm_ckf_pCorr_rnn] = correct(apbm_ckf_nonMk_rnn, [theta_bar_apbm_rnn; 1; 0; y], apbm_nn_rnn);
            switch n
                case 1
                    x_k_5_rnn = apbm_ckf_xCorr_rnn(end-3:end);
                case 2
                    x_k_4_rnn = apbm_ckf_xCorr_rnn(end-3:end);
                case 3
                    x_k_3_rnn = apbm_ckf_xCorr_rnn(end-3:end);
                case 4
                    x_k_2_rnn = apbm_ckf_xCorr_rnn(end-3:end);
                case 5
                    x_k_1_rnn = apbm_ckf_xCorr_rnn(end-3:end);
            end
        else
            [apbm_xPred_rnn, apbm_pPred_rnn] = predict(apbm_ckf_nonMk_rnn, Ts, apbm_nn_rnn, ...
                x_k_5_rnn, x_k_4_rnn, x_k_3_rnn, x_k_2_rnn, x_k_1_rnn);
            % correct with augmented likelihood function:
            [apbm_ckf_xCorr_rnn, apbm_ckf_pCorr_rnn] = correct(apbm_ckf_nonMk_rnn, [theta_bar_apbm_rnn; 1; 0; y], apbm_nn_rnn);
            % update memory
            x_k_5_rnn = x_k_4_rnn;
            x_k_4_rnn = x_k_3_rnn;
            x_k_3_rnn = x_k_2_rnn;
            x_k_2_rnn = x_k_1_rnn;
            x_k_1_rnn = apbm_ckf_xCorr_rnn(end-3:end);  
        end

        %% Pure NN CKF
        %NN CKF
        [nn_xPred, nn_pPred] = predict(nn_ckf, Ts, nn_mlp);
        % correct with augmented likelihood function:
        [nn_ckf_xCorr, nn_ckf_pCorr] = correct(nn_ckf, [theta_bar_nn ;y], nn_mlp);   

        % APBM MLP (Velocity as NN input)

        % testing/ making things flowing
%         P = apbm_ckf.StateCovariance;
%         if sum(isnan(P),'all') >= 1
%             disp(['P contains NANs! Reseting P'])
%             apbm_ckf.StateCovariance = P_apbm;
%             P = P_apbm;
%         end
%         if max(eig(P))> 1e4
%             disp(['max(eig(P))> 1e4 -> ', num2str(max(eig(P)))])
% %             apbm_ckf.State = zeros(size(apbm_ckf.State));
% %             apbm_ckf.StateCovariance = P_apbm;
%         end
%         min_eig = min(eig(P));
    
%         if min_eig < 1e-8
%             disp('here 1')
%             apbm_ckf.StateCovariance = apbm_ckf.StateCovariance  + 1*min_eig*eye(size(apbm_ckf.StateCovariance));
%         end
        P_old = P;
        % testing/ making things flowing
        P = nn_ckf.StateCovariance;
%         min_eig = min(eig(P));
%         if min_eig < 1e-4
%     %         disp('here')
%             nn_ckf.StateCovariance = nn_ckf.StateCovariance  + 1*min_eig*eye(size(nn_ckf.StateCovariance));
%         end


        % saving true states for plotting and error computations
        save_x(n,:) = [x(1),  x(3)]';
        save_Omega(n,:) = Omega;
        save_y(n,:) = y';

        % getting apbm nn params
        save_apbm_params_mlp(n,:) = apbm_ckf_xCorr_mlp(1:end-x_dim);
        apbm_ckf_xCorr_mlp = apbm_ckf_xCorr_mlp(end-x_dim+1:end);
        if n > nonMkStep
            save_apbm_params_nonMk(n,:) = apbm_ckf_xCorr_nonMk(1:end-x_dim);
            save_apbm_params_rnn(n,:) = apbm_ckf_xCorr_rnn(1:end-x_dim);
            % getting only states (not parameters)
            apbm_ckf_xCorr_nonMk = apbm_ckf_xCorr_nonMk(end-x_dim+1:end);
            apbm_ckf_xCorr_rnn = apbm_ckf_xCorr_rnn(end-x_dim+1:end);
        end

        % getting nn params
        save_nn_params(n,:) = nn_ckf_xCorr(1:end-nn_xdim);
        % getting only states (not parameters)
        nn_ckf_xCorr = nn_ckf_xCorr(end-nn_xdim+1:end);

        % saving ckf and nn_ckf estimated states
        save_ckf_x_mmse(n,:) = [ckf_xCorr(1),  ckf_xCorr(3)]';
        save_tm_ckf_x_mmse(n,:) = [ckf_tm_xCorr(1),  ckf_tm_xCorr(3)]';
        save_apbm_ckf_x_mmse_mlp(n,:) = [apbm_ckf_xCorr_mlp(1),  apbm_ckf_xCorr_mlp(3)]';
        if n > nonMkStep
            save_apbm_ckf_x_mmse_nonMk(n,:) = [apbm_ckf_xCorr_nonMk(1),  apbm_ckf_xCorr_nonMk(3)]';
            save_apbm_ckf_x_mmse_rnn(n,:) = [apbm_ckf_xCorr_rnn(1),  apbm_ckf_xCorr_rnn(3)]';
        end
        save_nn_ckf_x_mmse(n,:) = [nn_ckf_xCorr(1),  nn_ckf_xCorr(2)]';
        
        
    end
    
    save_x = save_x(nonMkStep+1:end,:);
    save_apbm_ckf_x_mmse_mlp = save_apbm_ckf_x_mmse_mlp(nonMkStep+1:end,:);
    save_apbm_ckf_x_mmse_nonMk = save_apbm_ckf_x_mmse_nonMk(nonMkStep+1:end,:);
    save_apbm_ckf_x_mmse_rnn = save_apbm_ckf_x_mmse_rnn(nonMkStep+1:end,:);
    save_nn_ckf_x_mmse = save_nn_ckf_x_mmse(nonMkStep+1:end,:);
    save_ckf_x_mmse = save_ckf_x_mmse(nonMkStep+1:end,:);
    save_tm_ckf_x_mmse = save_tm_ckf_x_mmse(nonMkStep+1:end,:);

    RMSE_APBM_CKF_MMSE_MLP = sqrt((norm(save_apbm_ckf_x_mmse_mlp - save_x).^2)/length(save_x))
    RMSE_APBM_CKF_MMSE_MLP_nonMk = sqrt((norm(save_apbm_ckf_x_mmse_nonMk - save_x).^2)/length(save_x))
    RMSE_APBM_CKF_MMSE_RNN = sqrt((norm(save_apbm_ckf_x_mmse_rnn - save_x).^2)/length(save_x))
    RMSE_NN_CKF_MMSE = sqrt((norm(save_nn_ckf_x_mmse - save_x).^2)/length(save_x))
    RMSE_CKF_MMSE = sqrt((norm(save_ckf_x_mmse - save_x).^2)/length(save_x))
    RMSE_CKF_TM_MMSE = sqrt((norm(save_tm_ckf_x_mmse - save_x).^2)/length(save_x))
    
    
    
    save_apbm_rmse_mlp(r) = RMSE_APBM_CKF_MMSE_MLP;
    save_apbm_rmse_mlp_nonMk(r) = RMSE_APBM_CKF_MMSE_MLP_nonMk; 
    save_apbm_rmse_rnn(r) = RMSE_APBM_CKF_MMSE_RNN; 
    save_cv_rmse(r) = RMSE_CKF_MMSE;    
    save_nn_rmse(r) = RMSE_NN_CKF_MMSE;    
    save_tm_rmse(r) = RMSE_CKF_TM_MMSE;
end
 
%% Plots
fontsize=16;
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

h0 = figure;
plot(save_x(:,1),save_x(:,2),'-*','LineWidth',.1), hold on, grid
% plot(save_y(:,1),save_y(:,2),'.','LineWidth',1)
plot(save_apbm_ckf_x_mmse_mlp(:,1),save_apbm_ckf_x_mmse_mlp(:,2), '-s','LineWidth',.1)
plot(save_apbm_ckf_x_mmse_nonMk(:,1),save_apbm_ckf_x_mmse_nonMk(:,2), '-s','LineWidth',.1)
plot(save_apbm_ckf_x_mmse_rnn(:,1),save_apbm_ckf_x_mmse_rnn(:,2), '-s','LineWidth',.1)
plot(save_nn_ckf_x_mmse(:,1),save_nn_ckf_x_mmse(:,2), '-s','LineWidth',.1)
plot(save_ckf_x_mmse(:,1),save_ckf_x_mmse(:,2), '-^','LineWidth',.1)
plot(save_tm_ckf_x_mmse(:,1),save_tm_ckf_x_mmse(:,2), '-o','LineWidth',.1)

% scatter(save_y(:,1), save_y(:,2))
scatter(0,0,'xk', 'linewidth',2)
ax = gca; ax.FontSize = fontsize-2;
xlabel('x [m]','fontsize', fontsize) 
ylabel('y [m]', 'fontsize', fontsize)
legend('True', 'APBM MLP MK','APBM MLP NMK','RNN','NN','PBM','TM','Sensor','Location','southeast', 'fontsize', fontsize-2)
% rectangle('Position',[-1 -1 2 2],'EdgeColor','k'), daspect([1 1 1])
% text(-2,2,'Sensor', 'fontsize', fontsize)

exportgraphics(h0, 'nonMkfigs_rnn/ctr_trajectrories.pdf')
%

RMSE_APBM_CKF_MMSE_MLP = sqrt((norm(save_apbm_ckf_x_mmse_mlp - save_x).^2)/length(save_x));
RMSE_APBM_CKF_MMSE_MLP_nonMk = sqrt((norm(save_apbm_ckf_x_mmse_nonMk - save_x).^2)/length(save_x));
RMSE_APBM_CKF_MMSE_RNN = sqrt((norm(save_apbm_ckf_x_mmse_rnn - save_x).^2)/length(save_x));
RMSE_NN_CKF_MMSE = sqrt((norm(save_nn_ckf_x_mmse - save_x).^2)/length(save_x));
RMSE_CKF_MMSE = sqrt((norm(save_ckf_x_mmse - save_x).^2)/length(save_x));
RMSE_CKF_TM_MMSE = sqrt((norm(save_tm_ckf_x_mmse - save_x).^2)/length(save_x));

%
h1 = figure;
A = [save_tm_rmse, save_apbm_rmse_mlp, save_apbm_rmse_mlp_nonMk, save_apbm_rmse_rnn, save_nn_rmse, save_cv_rmse];
A(A>1e5) = NaN;
boxchart(A)
ax = gca; ax.FontSize = fontsize-2;
xticklabels({'TM','APBM MLP MK','APBM MLP NMK','RNN','NN','PBM'})
ylabel('RMSE [m]')
grid
exportgraphics(h1, 'nonMkfigs_rnn/ctr_rmse_boxplots.pdf')


h2 = figure;
% boxchart(A(~isnan(save_apbm_rmse), :))
boxchart(A)
ylim([0,100])
ax = gca; ax.FontSize = fontsize-2;
xticklabels({'TM','APBM MLP MK','APBM MLP NMK','RNN','NN','PBM'})
ylabel('RMSE [m]')
grid
exportgraphics(h2, 'nonMkfigs_rnn/ctr_rmse_boxplots_zoom.pdf')
%

tvec = [0:Nt-4]*Ts;
h3=figure;
plot(tvec,sum(sqrt((save_tm_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',1), hold on, grid
plot(tvec,sum(sqrt((save_apbm_ckf_x_mmse_mlp - save_x).^2), 2),'-','LineWidth',1)
plot(tvec,sum(sqrt((save_apbm_ckf_x_mmse_nonMk - save_x).^2), 2),'-','LineWidth',1)
plot(tvec,sum(sqrt((save_apbm_ckf_x_mmse_rnn - save_x).^2), 2),'-','LineWidth',1)
plot(tvec,sum(sqrt((save_nn_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',1)
plot(tvec,sum(sqrt((save_ckf_x_mmse - save_x).^2), 2),'-','LineWidth',1)
xlabel('time [s]', 'fontsize', fontsize), 
ylabel('RMSE [m]', 'fontsize', fontsize)
legend('TM', 'APBM MLP MK','APBM MLP NMK','RNN','NN', 'PBM', 'fontsize', fontsize-2, 'location','best')
ax = gca; ax.FontSize = fontsize-2;

exportgraphics(h3, 'nonMkfigs_rnn/ctr_time_rmse.pdf')
%
% h4 = figure;
% plot(tvec, save_apbm_params_v_nonMk), grid
% xlabel('time [s]', 'fontsize', fontsize) 
% ylabel('\boldmath$\theta$', 'fontsize', fontsize)
% ax = gca; ax.FontSize = fontsize-2;
% exportgraphics(h4, 'nonMkfigs_rnn/ctr_param_evolution.pdf')

%
h5 = figure;
cdfplot(sum((save_tm_ckf_x_mmse - save_x).^2,2))
hold on
cdfplot(sum((save_apbm_ckf_x_mmse_mlp - save_x).^2,2))
cdfplot(sum((save_apbm_ckf_x_mmse_nonMk - save_x).^2,2))
cdfplot(sum((save_apbm_ckf_x_mmse_rnn - save_x).^2,2))
cdfplot(sum((save_nn_ckf_x_mmse - save_x).^2,2))
cdfplot(sum((save_ckf_x_mmse - save_x).^2,2))
legend('TM','APBM MLP MK','APBM MLP NMK','RNN','NN','PBM','fontsize', fontsize-2, 'location','best')
xlabel('squared error', 'fontsize', fontsize)
ylabel('CDF', 'fontsize', fontsize)
title('')
xlim([0,4000])
ylim([0.7,1.0])
ax = gca; ax.FontSize = fontsize-2;
exportgraphics(h5, 'nonMkfigs_rnn/ctr_cdf_squared_error.pdf')

%% Functions

function [x] = const_vel_transition_function(x_prev, Ts)
    F = [1 Ts 0 0;
         0 1 0 0;
         0 0 1 Ts;
         0 0 0 1];
     x = F*x_prev;
end

function [x] = cv_delay_feedback_function(x_prev, Ts, x_k_delta, ref)
    %x_prev = [s_prev, omega_prev]
    %x_k_delta = [s_k_delta, omega_k_delta]
    F = [1 Ts 0 0;
         0 1 0 0;
         0 0 1 Ts;
         0 0 0 1];
    omega_k_delta = x_k_delta(end);
    s_k_delta = x_k_delta(1:end-1);
    error = s_k_delta - ref;
    OTs = omega_k_delta*Ts;
    if omega_k_delta ==0
        G = eye(4);
    else
        G = [1, sin(OTs)/omega_k_delta, 0, -(1-cos(OTs))/omega_k_delta;
             0, cos(OTs), 0, -sin(OTs);
             0, (1-cos(OTs))/omega_k_delta, 1, sin(OTs)/omega_k_delta;
             0, sin(OTs), 0, cos(OTs)];
    end
    omega_prev = x_prev(end);
    s_prev = x_prev(1:end-1);
    B = -5*eye(4);

    s = F*s_prev + B*(sigmoid(G*error) - 0.5);
    omega = omega_prev;
    x = [s; omega];
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

function [x] = ctr_transition_function_nMk(x_prev, Ts, x_k_4, x_k_3, x_k_2)
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
    % non-Markovity
    s = B*s_prev;
    Omega = Omega_prev;
    if abs(x_k_4(1)) > 50 
        s(1) = s(1) - 4*Ts*(x_k_4(2) + x_k_3(2) + x_k_2(2))/3;
    end
    if abs(x_k_4(3)) > 100
        s(3) = s(3) - 4*Ts*(x_k_4(4) + x_k_3(4) + x_k_2(4))/3;
    end
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

function [x] = apbm_transition_function_nonMk(x_prev, Ts, nn_mlp, x_k_5, x_k_4, x_k_3, x_k_2, x_k_1)
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
    history = [x_k_5; x_k_4; x_k_3; x_k_2; x_k_1];
    s = w(1)*F*s + w(2)*nn_mlp.forward(history);
    x = [theta; w; s];
end

function [x] = apbm_transition_function_v(x_prev, Ts, nn_mlp, x_k_1, x_k_2, x_k_3, x_k_4) % use velocity as NN input 
    F = [1 Ts 0 0;
     0 1 0 0;
     0 0 1 Ts;
     0 0 0 1];
    theta = x_prev(1:nn_mlp.nparams);
    w = x_prev(nn_mlp.nparams + 1: nn_mlp.nparams + 2);
    s = x_prev(nn_mlp.nparams + 3: end);
    nn_mlp.set_params(theta)
    history = [x_k_1(end-2);x_k_1(end);x_k_2(end-2);x_k_2(end);x_k_3(end-2);x_k_3(end);x_k_4(end-2);x_k_4(end)];
    s = w(1)*F*s + w(2)*nn_mlp.forward(history);
    x = [theta; w; s];
end

function [x] = apbm_transition_function_rnn(x_prev, Ts, nn_rnn, x_k_5, x_k_4, x_k_3, x_k_2, x_k_1)
    F = [1 Ts 0 0;
     0 1 0 0;
     0 0 1 Ts;
     0 0 0 1];
    theta = x_prev(1:nn_rnn.nparams);
    w = x_prev(nn_rnn.nparams + 1: nn_rnn.nparams + 2);
    s = x_prev(nn_rnn.nparams + 3: end);
    nn_rnn.set_params(theta)
    history = [x_k_5, x_k_4, x_k_3, x_k_2, x_k_1];
    s = w(1)*F*s + w(2)*nn_rnn.forward(history);
    x = [theta; w; s];
end

function [x] = apbm_transition_function_rnn_v(x_prev, Ts, nn_rnn, x_k_1, x_k_2, x_k_3, x_k_4) % use velocity as NN input 
    F = [1 Ts 0 0;
     0 1 0 0;
     0 0 1 Ts;
     0 0 0 1];
    theta = x_prev(1:nn_rnn.nparams);
    w = x_prev(nn_rnn.nparams + 1: nn_rnn.nparams + 2);
    s = x_prev(nn_rnn.nparams + 3: end);
    nn_rnn.set_params(theta)
    history = [[x_k_1(end-2);x_k_1(end)],[x_k_2(end-2);x_k_2(end)], [x_k_3(end-2);x_k_3(end)], [x_k_4(end-2);x_k_4(end)]];
    s = w(1)*F*s + w(2)*nn_rnn.forward(history);
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


