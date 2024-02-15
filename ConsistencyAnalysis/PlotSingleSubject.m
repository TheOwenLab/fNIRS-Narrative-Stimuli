%% plot permutation stats results for comparisons 

% load in ISC results
ISCPerm = load("ISC.mat").ISC(1:4,:,:,:);

% average over HbO and HbR
ISCPerm = squeeze(mean(ISCPerm,3));

% load mask 
load("GroupFDRHO.mat")
Mask = GroupFDRHO; 

% initialize variable
lambda = 0.000005; 
SSlist = [8 29 52 66 75 92 112 125]; 

% load in table of channel names
cT = readtable("ChannelProjToCortex.xlsx");
Chans = 1:129; Chans(SSlist) = [];
nSub = 26; 
%#% plot results of each condition

%% BYD
counter = 1;
for ss = 1:nSub
    BYD = squeeze(ISCPerm(1,:,ss));
    BYD_Mask = logical(squeeze(Mask(1,:,ss)));
    % add back short channels to match the dimensions back with the MCS
    %initalize mask
    ISmask = nan(129,1);
    t = BYD;
    % set mask to nan
    t(~BYD_Mask) = nan;
    ISmask(Chans) = t; 
    limit = [-0.4,0.4];
    Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'turbo')
    sigRegions = cT{~isnan(ISmask),"LabelName"};
    disp(sigRegions)
    figure(counter)
    counter = counter + 2;
    saveas(gcf,string(ss) + '_GroupBYD.png')
end


%% TKN
close all
counter = 1;
for ss = 1:nSub
    TKN = squeeze(ISCPerm(3,:,ss));
    TKN_Mask = logical(squeeze(Mask(3,:,ss)));
    % add back short channels to match the dimensions back with the MCS
    %initalize mask
    ISmask = nan(129,1);
    t = TKN;
    % set mask to nan
    t(~TKN_Mask) = nan;
    ISmask(Chans) = t; 
    limit = [-0.4,0.4];
    Create_3D_Plot_projection_MC_Androu_2(ISmask, lambda,limit,SSlist,'turbo')
    sigRegions = cT{~isnan(ISmask),"LabelName"};
    disp(sigRegions)
    figure(counter)
    counter = counter + 2;
    saveas(gcf,string(ss) + '_GroupTKN.png')
end
