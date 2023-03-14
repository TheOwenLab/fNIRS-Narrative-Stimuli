%% plot feature derived from ML 

% initialize variables
lambda = 0.000005; limit = [0,1];
SSlist = [8 29 52 66 75 92 112 125]; nSub = 15;
oC = [62, 63, 64, 65, 118, 119, 120, 121];
% optional remove temporal channels
tC = [38 40 41 42 43 44 45 46 51 53 55 59 60 61, 96 98 99 100 101 102 106 107 109, 110, 114, 115, 116, 117];
% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");
Chans = 1:129; Chans(SSlist) = [];
%#% plot results of each condition

%% Taken Features
% load results 
NaciMask = load("TakenSingleChannelRes.mat").pAcc;
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = nan(129,1);
ISmask(Chans) = NaciMask;
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'jet')
sigRegions = cT{ISmask >= 0.8,"LabelName"};
disp(sigRegions)
figure(1)
exportgraphics(gcf,'TakenMLFeat.png','Resolution',300)

%% BYD Features
% load results 
NaciMask = load("BYDSingleChannelRes.mat").pAcc;
% add back short channels to match the dimensions back with the MCS
%initalize mask
ISmask = nan(129,1);
ISmask(Chans) = NaciMask;
%Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'jet')
sigRegions = cT{ISmask >= 0.8,"LabelName"};
disp(sigRegions)
figure(1)
exportgraphics(gcf,'BYDMLFeat.png','Resolution',300)