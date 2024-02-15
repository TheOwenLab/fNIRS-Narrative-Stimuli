%% plot permutation stats results for comparisons 

% load in ISC results
ISCPerm = load("GroupFDRT.mat").GroupFDRT;
% load p vals
load("GroupFDRBin.mat")
P = GroupFDRBin;

% load the uncorrected p values
load("GroupUncP.mat")
PUnc = GroupUncP;

% load q values
load("GroupFDRSum.mat")
Q = GroupFDRSum;

% initialize variable
lambda = 0.000005; 
SSlist = [8 29 52 66 75 92 112 125]; 

% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");
Chans = 1:129; Chans(SSlist) = [];

%#% plot results of each condition

%% BYD 
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
limit = [4,ceil(max(BYD,[],'all'))];
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'autumn')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(1)
saveas(gcf,'GroupBYD.png')

% save BYD mask
BYDMask = BYD_P; 
save("BYDFDR.mat","BYDMask");
save("BYDNamesFDR.mat","sigRegions");
cT = readtable("ChannelProjToCortex.xlsx");
ISmask = zeros(129,1);
ISmask(Chans) = t;
Pmask = ones(129,1);
Pmask(Chans) = BYD_P;
PUmask = ones(129,1);
PUmask(Chans) = PUnc(1,:);
Qmask = ones(129,1);
Qmask(Chans) = Q(1,:);
cT{:,"q"} = Qmask;
cT{:,"Z"} = ISmask;
%cT{:,"q"} = Pmask;
cT{:,"p"} = PUmask;
writetable(cT,"BYDResults.csv")

%% BYD Scrambled 
BYDS = squeeze(ISCPerm(2,:));
BYDS_P = squeeze(P(2,:));
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = nan(129,1);
Pmask = ones(129,1);
t = BYDS;
ISmask(Chans) = t; 
%ISmask(ISmask <= 0) = nan;
Pmask(Chans) = BYDS_P; 
Pmask(Pmask == 0) = nan;
ISmask(isnan(Pmask)) = nan;
% if plotting mask
%ISmask(~isnan(ISmask)) = 1; limit = [-1,1];
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'autumn')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(3)
saveas(gcf,'GroupBYDS.png')

% save BYD mask
BYDSMask = BYDS_P;
save("BYDSFDR.mat","BYDSMask");
save("BYDSNamesFDR.mat","sigRegions");
cT = readtable("ChannelProjToCortex.xlsx");
ISmask = zeros(129,1);
ISmask(Chans) = t;
Pmask = ones(129,1);
Pmask(Chans) = BYDS_P;
PUmask = ones(129,1);
PUmask(Chans) = PUnc(2,:);
Qmask = ones(129,1);
Qmask(Chans) = Q(2,:);
cT{:,"q"} = Qmask;
cT{:,"Z"} = ISmask;
%cT{:,"q"} = Pmask;
cT{:,"p"} = PUmask;
writetable(cT,"BYDSResults.csv")


%% TKN  
TKN = squeeze(ISCPerm(3,:));
TKN_P = squeeze(P(3,:));
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
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
% if plotting mask
%ISmask(~isnan(ISmask)) = 1; limit = [-1,1];
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'autumn')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(5)
saveas(gcf,'GroupTKN.png')

cT = readtable("ChannelProjToCortex.xlsx");
ISmask = zeros(129,1);
ISmask(Chans) = t;
Pmask = ones(129,1);
Pmask(Chans) = TKN_P; 
PUmask = ones(129,1);
PUmask(Chans) = PUnc(3,:);
Qmask = ones(129,1);
Qmask(Chans) = Q(3,:);
cT{:,"q"} = Qmask;
cT{:,"Z"} = ISmask;
%cT{:,"q"} = Pmask;
cT{:,"p"} = PUmask;
writetable(cT,"TakenResults.csv")
% save Taken mask
TakenMask = TKN_P;
save("TKNFDR.mat","TakenMask");
save("TKNNamesFDR.mat","sigRegions");

%% TKNS  
TKNS = squeeze(ISCPerm(4,:));
TKNS_P = squeeze(P(4,:));
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = nan(129,1);
Pmask = ones(129,1);
t = TKNS;
ISmask(Chans) = t;
%ISmask(ISmask <= 0) = nan;
Pmask(Chans) = TKNS_P; 
Pmask(Pmask == 0) = nan;
ISmask(isnan(Pmask)) = nan;
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
% if plotting mask
%ISmask(~isnan(ISmask)) = 1; limit = [-1,1];
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'autumn')
sigRegions = cT{~isnan(ISmask),"LabelName"};
disp(sigRegions)
figure(7)
saveas(gcf,'GroupTKNS.png')

cT = readtable("ChannelProjToCortex.xlsx");
ISmask = zeros(129,1);
ISmask(Chans) = t;
Pmask = ones(129,1);
Pmask(Chans) = TKNS_P; 
PUmask = ones(129,1);
PUmask(Chans) = PUnc(4,:);
Qmask = ones(129,1);
Qmask(Chans) = Q(4,:);
cT{:,"q"} = Qmask;
cT{:,"Z"} = ISmask;
%cT{:,"q"} = Pmask;
cT{:,"p"} = PUmask;
writetable(cT,"TakenSResults.csv")
% save Taken mask
TakenSMask = TKNS_P;
save("TKNSFDR.mat","TakenSMask");
save("TKNSNamesFDR.mat","sigRegions");



