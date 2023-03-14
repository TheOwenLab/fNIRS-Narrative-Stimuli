%% plot permutation stats results

% load in ISC results
ISCPerm = load("TGroupML.mat").ISCResAverageDiffG;

% load p vals
load("GroupFDRML.mat")
GroupP = GroupFDR;

cutoff = 0.05;

% initialize variable
lambda = 0.000005; limit = [2,ceil(max(ISCPerm,[],'all'))];
limit = [floor(min(ISCPerm,[],'all')),ceil(max(ISCPerm,[],'all'))];
SSlist = [8 29 52 66 75 92 112 125]; nSub = 15;
oC = [62, 63, 64, 65, 118, 119, 120, 121];
% optional remove temporal channels
tC = [38 40 41 42 43 44 45 46 51 53 55 59 60 61, 96 98 99 100 101 102 106 107 109, 110, 114, 115, 116, 117];
% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");
Chans = 1:129; Chans(SSlist) = [];
%#% plot results of each condition
%% intact BYD
close all
counter = 1;
for ss = 1:1
    %subplot(nSub,1,ss)
    ISCBYD = squeeze(ISCPerm(1,:,ss));
    % add back short channels to match the dimensions back with the MCS
    %initalize mask
    ISmask = nan(129,1);
    Pmask = ones(129,1);
    t = ISCBYD;
    % if masking
    %t(oC) = nan; t(tC) = nan;
    ISmask(Chans) = t;
    %ISmask(ISmask < 0) = nan;
    Pmask(Chans) = GroupP(1,:,ss); 
    Pmask(Pmask > cutoff) = nan;
    ISmask(isnan(Pmask)) = nan;
    %Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
    %Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'autumn')
    sigRegions = cT{~isnan(ISmask),"LabelName"};
    figure(counter);
    %savefig(string(["MaskConsistency_" + ss + ".fig"]))
    counter = counter + 2;
    disp(sigRegions)
end

%% Intact vs Scrambled
%% first BYD 
% load in ISC results
ISCPerm = load("ISMLp.mat").IS;
gBYDMask = [];
% load p vals
load("ISPFDRMLp.mat")
GroupP = ISPFDR;
for ss = 1:nSub
    %subplot(nSub,1,ss)
    IS = squeeze(ISCPerm(1,:,ss));
    % add back short channels to match the dimensions back with the MCS
    %initalize mask
    ISmask = nan(129,1);
    Pmask = ones(129,1);
    t = IS;
    % if masking
    %t(oC) = nan; t(tC) = nan;
    ISmask(Chans) = t;
    %ISmask(ISmask < 0) = nan;
    Pmask(Chans) = GroupP(1,:,ss); 
    Pmask(Pmask > cutoff) = nan;
    ISmask(isnan(Pmask)) = nan;
    %Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
    %Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'autumn')
    sigRegions = cT{~isnan(ISmask),"LabelName"};
    %figure(counter);
    %savefig(string(["MaskConsistency_" + ss + ".fig"]))
    %counter = counter + 2;
    disp(sigRegions)
    gBYDMask = [gBYDMask, ISmask];
end
% number of times regions appear across hold out sets 
nVals = sum(~isnan(gBYDMask),2); nValSig = nansum(sign(gBYDMask),2);
% set the sign
Create_3D_Plot_projection_MC_Androu_2(nVals, lambda,[1,nSub],SSlist,'autumn')
figure(1)
colormap(flipud(autumn))
exportgraphics(gcf,'BYDMaskFDRML.png','Resolution',300)
% save version for future masking
BYDMask = ~isnan(gBYDMask);
BYDMask(SSlist,:) = [];
save("BYDMaskFDRML.mat","BYDMask")
%% then Taken
% load in ISC results
ISCPerm = load("ISMLp.mat").IS;
gTKNMask = [];
% load p vals
load("ISPFDRMLp.mat")
GroupP = ISPFDR;
for ss = 1:nSub
    %subplot(nSub,1,ss)
    IS = squeeze(ISCPerm(2,:,ss));
    % add back short channels to match the dimensions back with the MCS
    %initalize mask
    ISmask = nan(129,1);
    Pmask = ones(129,1);
    t = IS;
    % if masking
    %t(oC) = nan; t(tC) = nan;
    ISmask(Chans) = t;
    %ISmask(ISmask < 0) = nan;
    Pmask(Chans) = GroupP(2,:,ss); 
    Pmask(Pmask > cutoff) = nan;
    ISmask(isnan(Pmask)) = nan;
    %Create_3D_Plot_Projection_Matt(ISmask, lambda,limit,[])
    %Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'autumn')
    sigRegions = cT{~isnan(ISmask),"LabelName"};
    %figure(counter);
    %savefig(string(["MaskConsistency_" + ss + ".fig"]))
    %counter = counter + 2;
    disp(sigRegions)
    gTKNMask = [gTKNMask, ISmask];
end
% number of times regions appear across hold out sets 
nVals = sum(~isnan(gTKNMask),2); nValSig = nansum(sign(gTKNMask),2);
Create_3D_Plot_projection_MC_Androu_2(nVals, lambda,[1,nSub],SSlist,'autumn')
figure(1)
colormap(flipud(autumn))
exportgraphics(gcf,'TakenMaskFDRML.png','Resolution',300)
% save version for future masking
TKNMask = ~isnan(gTKNMask);
TKNMask(SSlist,:) = [];
save("TakenMaskFDRML.mat","TKNMask")