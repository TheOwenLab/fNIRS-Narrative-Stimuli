%% plot permutation stats results for comparisons 

% load in ISC results
ISCPerm = load("ISp.mat").IS;
% load p vals
load("ISPFDRp.mat")
cutoff = 0.05;
ISP = ISPFDR;

% initialize variable
lambda = 0.000005; limit = [floor(min(ISCPerm,[],'all')),ceil(max(ISCPerm,[],'all'))];
SSlist = [8 29 52 66 75 92 112 125]; nSub = 15;
oC = [62, 63, 64, 65, 118, 119, 120, 121];
% optional remove temporal channels
tC = [38 40 41 42 43 44 45 46 51 53 55 59 60 61, 96 98 99 100 101 102 106 107 109, 110, 114, 115, 116, 117];
% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");
Chans = 1:129; Chans(SSlist) = [];

%#% plot results of each condition

%% BYD vs BYD S
BYD_BYDS = squeeze(ISCPerm(1,:));
BYD_BYDS_P = squeeze(ISP(1,:));
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = nan(129,1);
Pmask = ones(129,1);
t = BYD_BYDS;
% if masking
%t(oC) = nan; t(tC) = nan;
ISmask(Chans) = t; 
%ISmask(ISmask <= 0) = nan;
Pmask(Chans) = BYD_BYDS_P; 
Pmask(Pmask > cutoff) = nan;
ISmask(isnan(Pmask)) = nan;
% if plotting mask
%ISmask(~isnan(ISmask)) = 1; limit = [-1,1];
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'turbo')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(1)
saveas(gcf,'GroupBYDvBYDS.png')

% save BYD mask
BYDMask = BYD_BYDS_P < cutoff;
save("BYDvsScramMaskFDR.mat","BYDMask");
save("BYDvsScramNamesFDR.mat","sigRegions");
cT = readtable("ChannelProjToCortex.xlsx");
ISmask = zeros(129,1);
ISmask(Chans) = t;
Pmask = ones(129,1);
Pmask(Chans) = BYD_BYDS_P;
cT{:,"Z"} = ISmask;
cT{:,"P"} = Pmask;
writetable(cT,"BYDBYDSNEW.csv")

%% TKN vs TKNS 
TKN_TS = squeeze(ISCPerm(2,:));
TKN_TS_P = squeeze(ISP(2,:));
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = nan(129,1);
Pmask = ones(129,1);
t = TKN_TS;
% if masking
%t(oC) = nan; t(tC) = nan;
ISmask(Chans) = t;
%ISmask(ISmask <= 0) = nan;
Pmask(Chans) = TKN_TS_P; 
Pmask(Pmask > cutoff) = nan;
ISmask(isnan(Pmask)) = nan;
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
% if plotting mask
%ISmask(~isnan(ISmask)) = 1; limit = [-1,1];
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'turbo')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(3)
saveas(gcf,'GroupTKNvsTKNS.png')

cT = readtable("ChannelProjToCortex.xlsx");
ISmask = zeros(129,1);
ISmask(Chans) = t;
Pmask = ones(129,1);
Pmask(Chans) = TKN_TS_P; 
cT{:,"Z"} = ISmask;
cT{:,"P"} = Pmask;
writetable(cT,"TakenTakenSNEW.csv")
% save Taken mask
TakenMask = TKN_TS_P < cutoff;
save("TKNvsScramMaskFDR.mat","TakenMask");
save("TKNvsScramNamesFDR.mat","sigRegions");


