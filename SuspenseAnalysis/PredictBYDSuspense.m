% load in actual results
load("PreprocessedData.mat")

% put data in easy to use format
BYD = cat(4,data.BYD); BYDNoise = cat(4,data.BYDN);

% load in behaviour data
sPs = readtable("SuspenseBYD.csv");
sPs = sPs{:,"SUSP_BYD"};

% transform suspense ratings 
%sPs = normalize(sPs);

% resample Taken so that it matches with the 2.15 TR 
fs = 3.90625; 
% make resample values into integer
v = 100;

% optionally construct new time lagged and resampled timeseries 
[~,nChan,nHb,nSub] = size(BYD);
lagHR = 0;

% save lagged and resampled timeseries
rslBYD = nan(length(sPs),nChan,nHb,nSub); cCorT = nan(nChan,nHb,nSub);
rslBYDS = nan(length(sPs),nChan,nHb,nSub); cCorTS = nan(nChan,nHb,nSub);

for cc=1:nChan
    for ss = 1:nSub
        for hb = 1:nHb
            % grab channel values
            temp = BYD(:,cc,hb,ss);
            temps = BYDNoise(:,cc,hb,ss);

            % set resample parameters
            bll = length(temp); 
            TRl = (bll/fs)/length(sPs);
            bsll = length(temps); 
            TRsl = (bsll/fs)/length(sPs); 
            
            % resample
            trsl = resample(temp,v,fix(TRl * v *fs));
            tsrsl = resample(temps,v,fix(TRsl * v *fs));

            % trim final point to match same dimensions of sps
            % this may or may not be the case depending on how well
            % you can estimate the sampling frequency from two integers
            % required to  downsample the signal 
            rslBYD(:,cc,hb,ss) = trsl(1:end-1);
            rslBYDS(:,cc,hb,ss) = tsrsl(1:end-1);
            
        end
    end
end

%% collect bad channels
load("PreprocessedDataCWNIRS.mat")
BadChannels = zeros(nChan,nSub);
SSlist = [8 29 52 66 75 92 112 125]; 

% for each participant
for ss = 1:nSub
    % move from sc+lc space to lc space
    temp = nan(129,1);
    temp(gdata{ss}.SD.BadChannels) = 1;
    temp(SSlist) = [];
    BadChannels(temp == 1,ss) = 1;
end

clear gdata

%% stats for group averaged results
BS = cell(nChan,nHb); PS = cell(nChan,nHb);

% for each channel
for cc = 1:nChan
    % for each Hb
    % get chhanel info
    cinfo = BadChannels(cc,:);
    for hb = 1:nHb            
        % grab data values
        Ts = mean(rslBYD(:,cc,hb,~cinfo),4); Tss = mean(rslBYDS(:,cc,hb,~cinfo),4);
        % how well does suspense predict 
        [B,STATS] = robustfit([Tss,sPs],Ts); % note B is a 3x1 vector with the coefficients reflecting first the bias term, scrambled condition, then suspense ratings
        BS{cc,hb} = B; PS{cc,hb} = STATS;

    end

end
save("BYDSuspenseGroupBetas.mat","BS")

% unpack ps and ts and bs
Pso = [PS{:,1}]; Pso = [Pso.p];
Psr = [PS{:,2}]; Psr = [Psr.p];
Tso = [PS{:,1}]; Tso = [Tso.t];
Tsr = [PS{:,2}]; Tsr = [Tsr.t];
Bso = [BS{:,1}];
Bsr = [BS{:,2}]; 

% look at results for intact
Pgs = [Pso(end,:);Psr(end,:)]'; 
Tgs = [Tso(end,:);Tsr(end,:)]'; 

% fdr correction
[hois, ~,~,adj_p_ois] = fdr_bh(Pgs(:,1));
[hris, ~,~,adj_p_ris] = fdr_bh(Pgs(:,2));

GgP = sum([hois hris],2) > 1;
GgT = Tgs;
% store ts and ps
save("PredSuspenseGBYDt.mat","GgT")
save("PredSuspenseGBYDp.mat","GgP")

%% plot prediction of suspense statistics

% initialize plotting variables
lambda = 0.000005;
SSlist = [8 29 52 66 75 92 112 125]; 

% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");
Chans = 1:129; Chans(SSlist) = [];

% BYDvScrambled Correlations with Suspense

% load results 
load('PredSuspenseGBYDt.mat')
P = load("PredSuspenseGBYDp.mat").GgP;
Pmask = P;

% add back short channels to match the dimensions back with the MCS
ISmask = zeros(129,1);
cs = GgT(:,1);
cs(~Pmask) = 0;
ISmask(Chans) = cs;
% plot results
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,[0,5],SSlist,'jet')
sigRegions = cT{ISmask ~= 0,"LabelName"};
disp(sigRegions)
figure(1)
exportgraphics(gcf,'PredictBYD_Suspense_HbO.png','Resolution',300)

%% same for HbR
% load results 
load('PredSuspenseGBYDt.mat')
P = load("PredSuspenseGBYDp.mat").GgP;
Pmask = P;

% add back short channels to match the dimensions back with the MCS
ISmask = zeros(129,1);
cs = GgT(:,2);
cs(~Pmask) = 0;
ISmask(Chans) = cs;
% plot results
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,[-5,0],SSlist,'jet')
sigRegions = cT{ISmask ~= 0,"LabelName"};
disp(sigRegions)
figure(3)
exportgraphics(gcf,'PredictBYD_Suspense_HbR.png','Resolution',300)

% print stats
GgT(Pmask,:)
adj_p_ois(Pmask)
adj_p_ris(Pmask)
