%% correlate fp with behavioural measure

% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");

% load in actual results
load("PreprocessedData.mat")

% put data in easy to use format
Taken = cat(4,data.Taken); TakenNoise = cat(4,data.TakenNoise);

% drop none fp and significant channels
%Taken = Taken(:,TakenMask,:,:); TakenNoise = TakenNoise(:,TakenMask,:,:);

% load in behaviour data
sPs = readtable("SuspenseTaken.csv");
sPs = sPs{:,"Suspense_Z"};

% transform suspense ratings into something sensibly gaussian
%lsPs = log(sPs - min(sPs));
% infs were the minimum values, so replace with 0s
%lsPs(isinf(lsPs)) = 0;
sPs = normalize(sPs);

% resample Taken so that it matches with the 2.15 TR 
fs = 3.9063; 
% make resample values into integer
v = 100;

% optionally construct new time lagged and resampled timeseries 
[~,nChan,nHb,nSub] = size(Taken);
lagHR = 0;
% save lagged and resampled timeseries
rslT = nan(length(sPs),nChan,nHb,nSub); cCorT = nan(nChan,nHb,nSub);
rslTS = nan(length(sPs),nChan,nHb,nSub); cCorTS = nan(nChan,nHb,nSub);
for cc=1:nChan
    for ss = 1:nSub
        for hb = 1:nHb
            % grab channel values
            temp = Taken(:,cc,hb,ss);
            temps = TakenNoise(:,cc,hb,ss);
            % lag
            templagged = lagmatrix(temp,fix(fs * -lagHR));
            tempslagged = lagmatrix(temps,fix(fs * -lagHR));
            % drop nans
            templagged = templagged(~isnan(templagged));
            tempslagged = tempslagged(~isnan(tempslagged));
            tll = length(templagged); TRl = (tll/fs)/length(sPs); 
            tsll = length(tempslagged); TRsl = (tsll/fs)/length(sPs); 
            % resample
            trsl = resample(templagged,v,fix(TRl * v *fs));
            tsrsl = resample(tempslagged,v,fix(TRsl * v *fs));
            % trim final point to match same dimensions of sps
            % this may or may not be the case depending on how well
            % you can estimate the sampling frequency from two integers
            % required to  downsample the signal 
            rslT(:,cc,hb,ss) = trsl(1:end-1);
            rslTS(:,cc,hb,ss) = tsrsl(1:end-1);
            % estimate max cross correlation value
            %[~, cCorT(cc,hb,ss)] = max(xcorr(rslT(:,cc,hb,ss),sPs)); 
            %[~, cCorTS(cc,hb,ss)] = max(xcorr(rslTS(:,cc,hb,ss),sPs));
            %cCorT(cc,hb,ss)= cCorT(cc,hb,ss) - length(sPs);
            %cCorTS(cc,hb,ss) = cCorTS(cc,hb,ss)- length(sPs);
            
        end
    end
end

save("TakenResample.mat","rslT"); save("TNResample.mat",'rslTS')
%save("TKNCC.mat","cCorT"); save("TKNSCC.mat","cCorTS"); 
%% stats for all channels and participants
BS = cell(nChan,nHb,nSub); PS = cell(nChan,nHb,nSub);
bias = ones(length(sPs),1);

% for each channel
for cc = 1:nChan
    % for each subject
    for ss = 1:nSub
        % for each Hb
        for hb = 1:nHb            
            % grab data values
            Ts = rslT(:,cc,hb,ss); Tss = rslTS(:,cc,hb,ss);
            % how well does suspense predict 
            [B,STATS] = robustfit([Tss,sPs],Ts);
            BS{cc,hb,ss} = B; PS{cc,hb,ss} = STATS;
            
        end
    end
end
save("TakenSuspenseBetas.mat","BS")

% are group betas greater or less than zero
% for each channel
GP = nan(nChan,nHb); GT = nan(nChan,nHb);
for cc = 1:nChan
    % for each Hb
    for hb = 1:nHb 
        bs = [BS{cc,hb,:}];
        [H,P,CI,TVAL] = ttest(bs(end,:));
        GP(cc,hb) = P; GT(cc,hb) = TVAL.tstat;
    end
end

save("PredSuspenseTKNt.mat","GT")
save("PredSuspenseTKNp.mat","GP")

%% stats for group averaged results
BS = cell(nChan,nHb); PS = cell(nChan,nHb);
bias = ones(length(sPs),1);
rslTS = mean(rslTS,4); rslT = mean(rslT,4);
% for each channel
for cc = 1:nChan
    % for each Hb
    for hb = 1:nHb            
        % grab data values
        Ts = rslT(:,cc,hb); Tss = rslTS(:,cc,hb);
        % how well does suspense predict 
        [B,STATS] = robustfit([Tss,sPs],Ts);
        BS{cc,hb} = B; PS{cc,hb} = STATS;

    end

end
save("TakenSuspenseGroupBetas.mat","BS")
% unpack ps and ts and bs
Pso = [PS{:,1}]; Pso = [Pso.p];
Psr = [PS{:,2}]; Psr = [Psr.p];
Tso = [PS{:,1}]; Tso = [Tso.t];
Tsr = [PS{:,2}]; Tsr = [Tsr.t];
Bso = [BS{:,1}];
Bsr = [BS{:,2}]; 

% look at results for intact
Ps = [Pso(end,:);Psr(end,:)]'; 
Ts = [Tso(end,:);Tsr(end,:)]'; 
% fdr correction
[hois, ~,~,adj_p_ois] = fdr_bh(Pso);
[hris, ~,~,adj_p_ris] = fdr_bh(Psr);

GgP = [hois(end,:) & hris(end,:)]';
GgT = Ts;
% store ts and ps
save("PredSuspenseGTKNt.mat","GgT")
save("PredSuspenseGTKNp.mat","GgP")
%% plot prediction of suspense statistics
% initialize variable
lambda = 0.000005;
SSlist = [8 29 52 66 75 92 112 125]; nSub = 15;
oC = [62, 63, 64, 65, 118, 119, 120, 121];
% optional remove temporal channels
tC = [38 40 41 42 43 44 45 46 51 53 55 59 60 61, 96 98 99 100 101 102 106 107 109, 110, 114, 115, 116, 117];
% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");
Chans = 1:129; Chans(SSlist) = [];
%#% plot results of each condition

%% TKNvScrambled Correlations with Suspense

% load results 
load('PredSuspenseGTKNt.mat')
P = load("PredSuspenseGTKNp.mat").GgP;
%Pmask = (P(:,1) < 0.05 & P(:,2) < 0.05);
Pmask = P;
% load mask
load('TKNvsScramMaskFDR.mat')
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = zeros(129,1);
%TakenMask(oC) = 0; TakenMask(tC) = 0;
%mask = double(TakenMask);
%mask(TakenMask) = corrResGroupN(:,1);
cs = GgT(:,1);
cs(~Pmask) = 0;
ISmask(Chans) = cs;
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,[-5,5],SSlist,'jet')
sigRegions = cT{ISmask ~= 0,"LabelName"};
disp(sigRegions)
figure(1)
exportgraphics(gcf,'PredictTKN_Suspense_HbO.png','Resolution',300)

%% same for HbR
% load results 
load('PredSuspenseGTKNt.mat')
P = load("PredSuspenseGTKNp.mat").GgP;
%Pmask = (P(:,1) < 0.05 & P(:,2) < 0.05);
Pmask = P;
% load mask
load('TKNvsScramMaskFDR.mat')
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = zeros(129,1);
%TakenMask(oC) = 0; TakenMask(tC) = 0;
%mask = double(TakenMask);
%mask(TakenMask) = corrResGroupN(:,1);
cs = GgT(:,2);
cs(~Pmask) = 0;
ISmask(Chans) = cs;
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,[-5,5],SSlist,'jet')
sigRegions = cT{ISmask ~= 0,"LabelName"};
disp(sigRegions)
figure(3)
exportgraphics(gcf,'PredictTKN_Suspense_HbR.png','Resolution',300)

% print stats
disp(sigRegions)
GgT(Pmask,:)
adj_p_ois(end,Pmask)
adj_p_ris(end,Pmask)