%% plot permutation stats results for comparisons 

% load in ISC results
ISCPerm = load("GroupFDRTIgS.mat").GroupFDRTIgS;
% load p vals
load("GroupFDRBinIgS.mat")
P = GroupFDRBinIgS;

% load the uncorrected p values
load("GroupUncPIS.mat")
PUnc = GroupUncPIS;

% load q values
load("GroupFDRSumIgS.mat")
Q = GroupFDRSumIgS;

% initialize variable
lambda = 0.000005; 
SSlist = [8 29 52 66 75 92 112 125]; 

% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");
Chans = 1:129; Chans(SSlist) = [];

%#% plot results of each condition

%% BYD vs BYDS 
BYD = squeeze(ISCPerm(1,:));
BYD_P = squeeze(P(1,:));
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = nan(129,1);
Pmask = ones(129,1);
t = BYD;
ISmask(Chans) = t; 
%ISmask(ISmask <= 0) = nan;
Pmask(Chans) = BYD_P; 
Pmask(Pmask == 0) = nan;
ISmask(isnan(Pmask)) = nan;
% if plotting mask
%ISmask(~isnan(ISmask)) = 1; limit = [-1,1];
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
limit = [4,9];
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'autumn')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(1)
saveas(gcf,'GroupBYDgBYDS.png')

% save BYD mask
BYDvBYDSMask = BYD_P; 
save("BYDgBYDSFDR.mat","BYDvBYDSMask");
save("BYDgBYDSNamesFDR.mat","sigRegions");
cT = readtable("ChannelProjToCortex.xlsx");
ISmask = zeros(129,1);
ISmask(Chans) = t;
Pmask = ones(129,1);
Pmask(Chans) = BYD_P;

Qmask = ones(129,1);
Qmask(Chans) = Q(1,:);
cT{:,"q"} = Qmask;
cT{:,"Z"} = ISmask;
cT{:,"P"} = Pmask;
writetable(cT,"BYDgBYDSResults.csv")

%% Taken 
TKN = squeeze(ISCPerm(2,:));
TKN_P = squeeze(P(2,:));
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = nan(129,1);
Pmask = ones(129,1);
t = TKN;
ISmask(Chans) = t; 
%ISmask(ISmask <= 0) = nan;
Pmask(Chans) = TKN_P; 
Pmask(Pmask == 0) = nan;
ISmask(isnan(Pmask)) = nan;
% if plotting mask
%ISmask(~isnan(ISmask)) = 1; limit = [-1,1];
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
limit = [4,9];
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'autumn')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(3)
saveas(gcf,'GroupTKNgTKNS.png')

% save BYD mask
TKNgTKNSMask = TKN_P;
save("TKNgTKNSFDR.mat","TKNgTKNSMask");
save("TKNgTKNSNamesFDR.mat","sigRegions");
cT = readtable("ChannelProjToCortex.xlsx");
ISmask = zeros(129,1);
ISmask(Chans) = t;
Pmask = ones(129,1);
Pmask(Chans) = TKN_P;

Qmask = ones(129,1);
Qmask(Chans) = Q(2,:);
cT{:,"q"} = Qmask;
cT{:,"Z"} = ISmask;
cT{:,"P"} = Pmask;
writetable(cT,"TKNgTKNSResults.csv")
