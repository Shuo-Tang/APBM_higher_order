    % Data generator and predictors (model based, AI data-based, APBM)
% - 1D position model - random walk model with measured position increment
% - for APBM concept illustration 
% - joint paper "AI-based Modelling for Navigation and Tracking"

clear; close;

% Initialisation
% - number of simulated data
N = 1e7;
% - initial position PDF
pinit_mean = 0;
pinit_var = 10^2;
% - noise properties
% -- position white noise variance
Q = 2^2;
% -- position increment measurement bias driving noise variance (time constant is varying and defined later)
Sigma2 = 1e-1^2;
% -- relation between time constant and independent variable
tauFunct = @(theta) exp(-1./(1e2*(theta-14.995)));
% - allocation
% -- true position, position increment, measured increment, bias, bias time constant, independent variable (e.g., temperature)
p = zeros(1,N);
u = zeros(1,N);
u_measured = zeros(1,N);
b = zeros(1,N);
tau = zeros(1,N);
theta = zeros(1,N);
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
for k = 1:N
    % - true position increment
    u(k) = sin(1+0.1*randn);
    % - independent variable value (e.g., temperature - between 15 and 25 degC)
    theta(k) = 20 + 5*cos(k*1e-3); % 12, 20, 120
    %theta(k) = 20; % 12, 20, 120
    % - dependent variable valu (i.e., time constant)
    tau(k) = tauFunct(theta(k));
    % - position
    w = sqrt(Q)*randn; % position noise
    p(k+1) = p(k) + u(k) + w; % position dynamics
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
R = 0;
% Pp = diag([10, 1]);
xp = zeros(2,N);
xf = zeros(2,N);
PpSteady = dare(F',H',Qx,R);
KSteady = (PpSteady*H')/(H*PpSteady*H'+R);
PfSteady = PpSteady - KSteady*H*PpSteady;
for k = 1:N-2
    % ------
    % one-step (approx 1 - neglecting w(k) and b(k))
    % ------
    % - position prediction
    p_mb1(k+1) = p(k) + u_measured(k) - b_mb1(k);
    % - bias prediction (zero - depending on the initial condition)
    b_mb1(k+1) = meanTau*b_mb1(k);
    % ------
    % one-step (approx 2 - neglecting w(k) and w(k-1))
    % ------
    % - position prediction
    if k>=3
        p_mb2(k+1) = (1+meanTau)*p(k) - meanTau*p(k-1) + u_measured(k) - meanTau*u_measured(k-1);
    end
    % ------
    % - one-step (opt KF-based - up to exactly unknown time constant)
    % ------
    %K = Pp*H'/(H*Pp*H'+R);
    zp = H*xp(:,k);
    xf(:,k) = xp(:,k) + KSteady*(p(k)-zp);
    %Pf = Pp - K*H*Pp;
    % --
    xp(:,k+1) = F*xf(:,k) + [u_measured(k); 0];
    %Pp = F*Pf*F' + Qx;
    % --
    p_mb3opt(k+1) = xp(1,k+1);
end
MSE_mb1 = var(p(1:N-1)-p_mb1(1:N-1));
MSE_mb2 = var(p(3:N-1)-p_mb2(3:N-1));
MSE_mb3opt = var(p(1:N-1)-p_mb3opt(1:N-1)); % should be same as PpSteady(1,1)

% plots and disps
% disp(['MB Prediction MSE: 1-step Approx1, 1-step Approx2, 1-step Op: ', num2str([MSE_mb1, MSE_mb2, MSE_mb3opt])])
table([MSE_mb1, MSE_mb2, MSE_mb3opt]','RowNames',{'1-step Approx1', '1-step Approx2', '1-step Optimal'},'VariableNames',{'MB Prediction MSE:'})
% disp(['Steady state position variance and respective MSE (should be same): ', num2str([PpSteady(1,1), MSE_mb3opt])])
disp('Steady state position variance and respective MSE')
table([PpSteady(1,1) MSE_mb3opt]','RowNames',{'variance','MSE'},'VariableNames',{'value'})
% t.Properties.Description = 'Comparison of stead state variance and MSE';
% --

figure
subplot(1,3,1)
plot(theta(1:2e4))
title('Independently varying parameter - temeperature')
xlabel('k')
ylabel('\theta')
subplot(1,3,2)
plot(tau(1:2e4))
title('Dependently varying bias time constant')
xlabel('k')
ylabel('\tau')
subplot(1,3,3)
plot(b)
hold
plot(xf(2,:))
title('Bias')
xlabel('k')
ylabel('bias')
legend('true','KF estim')
