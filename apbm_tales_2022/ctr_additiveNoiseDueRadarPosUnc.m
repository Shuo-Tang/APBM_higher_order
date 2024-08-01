% measurement eq. linearisation for non-additive measurement noise
% - Stirling interpolation, differences instead of derivatives

% state estimate (posx, velx, posy, vely)
% xpred = [100; 1; 200; 0.5];
% xpred = [10; 1; 20; 0.5];
xpred = [1; 1; 2; 0.5];

% radar position uncertainty xi (assuming radar position [0, 0])
SigmaRadarLoc = [1 0.2; 0.2 1.8];

% nonlinear function
hfun = @(x) [30 - 10*log10(norm(-x(1:2:3))^2.2); atan2(x(3),x(1))];
% hfun = @(x) x(1:2:3);

% nonlinear function linearisation wrt non-additive noise "xi" (with mean "ximean")
% z = h(x,xi) + v ~ h(x,xiMean) + J(x,ximean)(xi-ximean) + v

% *** Taylor expansion/Stirling' interpolation absed integration ***
% Approximate "Jacobian" calculation (evaluated at xpred and non-additive noise mean)
% - first order central differences
% - dh1/dxi1, dh2/dxi1
deltaxi = [sqrt(SigmaRadarLoc(1,1)); 0];
auxPosPlus = zeros(4,1);
auxPosMinus = zeros(4,1);
auxPosPlus(1) = xpred(1) + deltaxi(1);
auxPosPlus(3) = xpred(3) + deltaxi(2);
auxPosMinus(1) = xpred(1) - deltaxi(1);
auxPosMinus(3) = xpred(3) - deltaxi(2);
auxhPlus = hfun(auxPosPlus); 
auxhMinus = hfun(auxPosMinus); 
J(1,1) = (auxhPlus(1)-auxhMinus(1))/(2*deltaxi(1));
J(2,1) = (auxhPlus(2)-auxhMinus(2))/(2*deltaxi(1));
% - dh1/dxi2, dh2/dxi2
deltaxi = [0; sqrt(SigmaRadarLoc(2,2))];
auxPosPlus = zeros(4,1);
auxPosMinus = zeros(4,1);
auxPosPlus(1) = xpred(1) + deltaxi(1);
auxPosPlus(3) = xpred(3) + deltaxi(2);
auxPosMinus(1) = xpred(1) - deltaxi(1);
auxPosMinus(3) = xpred(3) - deltaxi(2);
auxhPlus = hfun(auxPosPlus); 
auxhMinus = hfun(auxPosMinus); 
J(1,2) = (auxhPlus(1)-auxhMinus(1))/(2*deltaxi(2));
J(2,2) = (auxhPlus(2)-auxhMinus(2))/(2*deltaxi(2));

% moments calculation
addMeasNoiseMeanSI = zeros(2,1);
addMeasNoiseCovSI = J*SigmaRadarLoc*J';

% *** cubature integration ***
% sigma points in radar position uncertainty (CKF sigma point set definition)
chix = [xpred(1); xpred(3)] + sqrt(2)*chol(SigmaRadarLoc)'*[1 0 -1 0; 0 1 0 -1];
ns = 4;
chi_weight = 1/ns*ones(1,ns);
% transformed sigma points (affected by uncertainty in range and bearing due to SigmaRadarLoc)
chiz = zeros(2,ns);
for idx=1:ns
    auxPos = zeros(4,1);
    auxPos(1) = chix(1,idx);
    auxPos(3) = chix(2,idx);
    chiz(:,idx) = hfun(auxPos);
end

% mean and covariance matrix in range and bearing due to SigmaRadarLoc
addMeasNoiseMeanCI = zeros(2,1);
addMeasNoiseCovCI = zeros(2,2);
for idx=1:ns
    addMeasNoiseMeanCI = addMeasNoiseMeanCI + chi_weight(idx)*chiz(:,idx);
end
for idx=1:ns
    addMeasNoiseCovCI = addMeasNoiseCovCI + chi_weight(idx)*(chiz(:,idx)-addMeasNoiseMeanCI)*(chiz(:,idx)-addMeasNoiseMeanCI)';
end
addMeasNoiseMeanCI = addMeasNoiseMeanCI - hfun(xpred);


% disps/plots
disp('Contribution of nonadditive noise - mean (CR, SI1/TE1):')
disp([addMeasNoiseMeanCI, addMeasNoiseMeanSI])
disp('Contribution of nonadditive noise - cov (CR, SI1/TE1):')
disp([addMeasNoiseCovCI, addMeasNoiseCovSI])