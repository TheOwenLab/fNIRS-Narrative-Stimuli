%% run t-space similarity 
% generate channel mask based on group results
% what type of correction
cor = "FDR"; nFA = 3; 
% load in mask
% if FDR correction
if cor == "FDR"
    load('TakenMaskFDRML.mat')
    load('BYDMaskFDRML.mat')
else
    load('TakenMaskML.mat')
    load('BYDMaskML.mat')
end

% drop channels in mask that could be due to chance
sigChanBYD = sum(BYDMask,2); sigChanTKN = sum(TKNMask,2);
BYDMask(sigChanBYD <= nFA,:) = 0; TKNMask(sigChanTKN <= nFA,:) = 0;
% combine masks based on conditions
Masks = {BYDMask,BYDMask,TKNMask,TKNMask};
% loop parameters
nSub = 15; nCond = 4; nChan = 121;
SubList = 1:nSub; c = 1;

%% compare consistency between subject and the group on dataset without the subject
% initalize matrix
simMatMask = nan(nSub, nCond, nSub - 1);
corMatMask = nan(nSub, nCond, nSub - 1);
ttestMatMask = nan(nSub, nCond, nSub - 1);
% load in datasets holding out subjects
load("TSubML.mat")
ISCML = ISCResAverageT(1:4,:,:,:);
load("TSub.mat")
ISCt = ISCResAverageT(1:4,:,:);
% for each subject
for ss = 1:nSub 
    % for each condition
    for nc = 1:nCond
        % what is the current mask
        mask = Masks{nc}(:,ss);
        % get participant data
        v1 = ISCt(nc,mask,ss); 
        % for each pairwise association
        for ls = 1:nSub - 1
            % get current subject's data
            v2 = ISCML(nc,mask,ss,ls);
            simMatMask(ss,nc,ls) = norm(v1 - v2)/sum(mask); % normalize the difference value
            corMatMask(ss,nc,ls) = corr(v1',v2'); % get the correlation value
            [H,P,CI,STATS] = ttest(v1,v2); % get a t-test
            ttestMatMask(ss,nc,ls) = STATS.tstat;
        end
    end
end


%% save results
save("tSpaceInd.mat","simMatMask");
save("ttestSpaceInd.mat","ttestMatMask");
save("corSpaceInd.mat","corMatMask");


