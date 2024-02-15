%% correlate fp with behavioural measure

% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");

% load in actual results
load("PreprocessedData.mat")

% put data in easy to use format
Taken = cat(4,data.Taken); TakenNoise = cat(4,data.TakenNoise);

% load in behaviour data
sPs = readtable("SuspenseTaken.csv");
sPs = sPs{:,"Suspense_Z"};

% transform suspense ratings 
%sPs = normalize(sPs);

% resample Taken so that it matches with the 2.15 TR 
fs = 3.90625; 
% make resample values into integer
v = 100;

% resample timeseries
[~,nChan,nHb,nSub] = size(Taken);

% save resampled timeseries
rslT = nan(length(sPs),nChan,nHb,nSub); 
rslTS = nan(length(sPs),nChan,nHb,nSub); 

% for each channel
for cc=1:nChan
    for ss = 1:nSub
        for hb = 1:nHb
            % grab channel values
            temp = Taken(:,cc,hb,ss);
            temps = TakenNoise(:,cc,hb,ss);
           
            % set resampling paramaters
            tll = length(temp); TRl = (tll/fs)/length(sPs); 
            tsll = length(temps); TRsl = (tsll/fs)/length(sPs);

            % resample
            trsl = resample(temp,v,fix(TRl * v *fs));
            tsrsl = resample(temps,v,fix(TRsl * v *fs));

            % trim final point to match same dimensions of sps
            % this may or may not be the case depending on how well
            % you can estimate the sampling frequency from two integers
            % required to  downsample the signal 
            rslT(:,cc,hb,ss) = trsl(1:end-1);
            rslTS(:,cc,hb,ss) = tsrsl(1:end-1);
            
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
    % get chhanel info
    cinfo = BadChannels(cc,:);
    % for each Hb
    for hb = 1:nHb            
        % grab data values
        Ts = mean(rslT(:,cc,hb,~cinfo),4); Tss = mean(rslTS(:,cc,hb,~cinfo),4);
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
TGs = [Tso(end,:);Tsr(end,:)]'; 
% fdr correction
[hois, ~,~,adj_p_ois] = fdr_bh(Ps(:,1));
[hris, ~,~,adj_p_ris] = fdr_bh(Ps(:,2));

GgP = sum([hois hris],2) > 1;
GgT = TGs;

% store ts and ps
save("PredSuspenseGTKNt.mat","GgT")
save("PredSuspenseGTKNp.mat","GgP")

%% plot prediction of suspense statistics
% initialize variable
lambda = 0.000005;
SSlist = [8 29 52 66 75 92 112 125];
% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");
Chans = 1:129; Chans(SSlist) = [];
%#% plot results of each condition

% TKNvScrambled Correlations with Suspense

% load results 
load('PredSuspenseGTKNt.mat')
P = load("PredSuspenseGTKNp.mat").GgP;
Pmask = P;

% add back short channels to match the dimensions back with the MCS
ISmask = nan(129,1);
cs = GgT(:,1);
cs(~Pmask) = nan;
ISmask(Chans) = cs;

% plot results
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,[0,5],SSlist,'jet')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(1)
exportgraphics(gcf,'PredictTKN_Suspense_HbO.png','Resolution',300)

%% same for HbR
% load results 
load('PredSuspenseGTKNt.mat')
P = load("PredSuspenseGTKNp.mat").GgP;
Pmask = P;

% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = nan(129,1);
cs = GgT(:,2);
cs(~Pmask) = nan;
ISmask(Chans) = cs;
% plot results
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,[-5,0],SSlist,'jet')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(3)
exportgraphics(gcf,'PredictTKN_Suspense_HbR.png','Resolution',300)

% print stats
disp(sigRegions)
GgT(Pmask,:)
adj_p_ois(Pmask)
adj_p_ris(Pmask)