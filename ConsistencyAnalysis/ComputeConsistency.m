%% compute normalized dot product and run basic statistics

% load in mask
load('GroupBinHOIgS.mat') % IgS hold out masks
Mask = [GroupBinHOIgS(1,:,:);GroupBinHOIgS(1,:,:);GroupBinHOIgS(2,:,:);GroupBinHOIgS(2,:,:)];

% load('GroupFDRHO.mat') % Condition Masks
% Mask = [GroupFDRHO(1,:,:);GroupFDRHO(1,:,:);GroupFDRHO(2,:,:);GroupFDRHO(2,:,:)];

% loop parameters
nSub = 26; nCond = 4; nChan = 121;
SubList = 1:nSub; c = 1;


%% measure consistency to the group INTACT mask
load("ISC.mat")
load("GroupAvHO.mat")

% replace GroupAv to only have intact values
GroupAvHO(2,:,:) = GroupAvHO(1,:,:);
GroupAvHO(4,:,:) = GroupAvHO(3,:,:);

ISCt = ISC;
simMatGroup = nan(nSub, nCond); simMatT = nan(nSub, nCond);

% for each subject
for ss = 1:nSub 
    
    % for each condition
    for nc = 1:nCond
        
        % get condition mask
        mask = logical(Mask(nc,:,ss));
        
        % get actual participant data
        v1 = mean(ISCt(nc,mask,:,ss),3); 
        
        % contrast with group
        % get the held out group data
        v2 = GroupAvHO(nc,mask,ss);
        
        % compute normalized dot product
        simMatGroup(ss,nc) = dot(v1,v2)/sum(mask);

    end
end


%% normalized dot product computed with phase surrogate data 
load('HoldOutNull.mat')
[nCond, nChan, nSub,nSubList,nIter] = size(HoldOutNull);
simMatGroupPerm = nan(nSub,nSubList,nCond,nIter); 

% for each iteration
for nn = 1:nIter
    % for each subject
    for ss = 1:nSub
        % for each condition
        for nc = 1:nCond
            % get condition mask
            mask = logical(Mask(nc,:,ss));
            % get the held out group data
            v2 = GroupAvHO(nc,mask,ss);
            
            % get permuted pariticpant data 
            for rs = 1:nSubList
                v1 = squeeze(HoldOutNull(nc,mask,ss,rs,nn));
                simMatGroupPerm(ss,rs,nc,nn) = dot(v1,v2)/sum(mask);
            end
        end
    end
end


%% results

% sanity check -- see if there are statistical differences in normalized dot product 
% between intact and scrambled conditions
[~,Pb,~,STATSb] = ttest(simMatGroup(:,1),simMatGroup(:,2));
[~,Pt,~,STATSt] = ttest(simMatGroup(:,3),simMatGroup(:,4));

% average null distribution of normalized dot product for each held-out subject 
simMatGroupPermMean = squeeze(mean(simMatGroupPerm,2));

counter = 0; Cond = 3; nDiff = 0; Q = [];

for ss = 1:nSub
    counter = counter + 1;
    figure(counter)
    histogram(simMatGroupPermMean(ss,Cond,:))
    hold on
    xline(simMatGroup(ss,Cond))
    hold off
    q = (sum(simMatGroup(ss,Cond) < simMatGroupPermMean(ss,Cond,:)))/nIter;
    q 
    Q = [Q, q];
    if q > (0.05/26)
        nDiff = nDiff + 1;
    end
end
(nSub - nDiff)

% what about fdr correction?
[~,~,~,QFDR] = fdr_bh(Q);
sum(QFDR < 0.05)


save("NormalizedDotProduct.mat","simMatGroup")
