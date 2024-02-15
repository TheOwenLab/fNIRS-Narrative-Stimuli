%% Conduct Permuatation Statistics

%% Compute FDR within each left out dataset
% load in data 
load("ISCHoldOut.mat"); load("MaxTHoldOut.mat");
ISC = squeeze(nanmean(ISC,3));

[nCond,nChan,nSub,nSubList] = size(ISC);

GroupTHO = nan(nCond,nChan,nSub); GroupAvHO = nan(nCond,nChan,nSub);
GroupPHO = nan(nCond,nChan,nSub); GroupFDRHO = nan(nCond,nChan,nSub);
% for each condition
for nc = 1:nCond
    % for each left out subject.
    tstar = prctile(MaxTHoldOut(nc,:),95)
    for ss = 1:nSub
        % for each channel 
        for cc = 1:nChan
            % actual value
            aISC = squeeze(ISC(nc,cc,ss,:));
            [H,P,CI,ZVAL] = ttest(aISC,0,'tail','right'); % just a ttest
            GroupTHO(nc,cc,ss) = ZVAL.tstat;
            GroupFDRHO(nc,cc,ss) = ZVAL.tstat > tstar;
            % store group level p-vals
            GroupPHO(nc,cc,ss) = P;

        end
        % grab average values
        GroupAvHO(nc,:,ss) = nanmean(squeeze(ISC(nc,:,ss,:)),2);

    end
   
end
save("GroupTHO.mat","GroupTHO");
save("GroupPHO.mat","GroupPHO");
save("GroupFDRHO.mat","GroupFDRHO");
save("GroupAvHO.mat","GroupAvHO")

%% now for intact > scrambled (STILL WORKING ON THIS)
% load in data 
load("ISCHoldOut.mat"); load("MaxTHoldOutIS.mat");

[nCond,nChan,~,nSub,nSubList] = size(ISC);

nIter = size(MaxTHoldOutIS,2);
GroupBinHOIgS = nan(nCond/2,nChan,nSub); GroupFDRHOIgS = nan(nCond/2,nChan,nSub);
GroupTHOIgS = nan(nCond/2,nChan,nSub); GroupUncPHOIgS = nan(nCond/2,nChan,nSub);
GroupAvHOIgS = nan(nCond/2,nChan,nSub);

% for each condition
counter = 0;
for nc = 1:2:nCond
   counter = counter + 1;
   % for each left out subject.
    tstar = prctile(MaxTHoldOutIS(counter,:),95)
    for ss = 1:nSub
        % actual values
        aISCi = squeeze(ISC(nc,:,:,ss,:));
        aISCs = squeeze(ISC(nc+1,:,:,ss,:));
        rdiff = squeeze((aISCi(:,1,:) - aISCs(:,1,:)) + (aISCi(:,2,:) - aISCs(:,2,:)));
        rdiff = rdiff/2;
        for cc = 1:nChan
            % generate test statistic
            [~,GroupUncPHOIgS(nc,cc,ss),~,STATS] = ttest(rdiff(cc,:),0,'tail','right');
            % store group level statistics
            GroupTHOIgS(counter,cc,ss) =  STATS.tstat;
            GroupBinHOIgS(counter,cc,ss) = STATS.tstat > tstar;
            GroupFDRHOIgS(counter,cc,ss) = sum(STATS.tstat < MaxTHoldOutIS(counter,:))/nIter;
        end
        GroupAvHOIgS(counter,:,ss) = nanmean(rdiff,2);
    end
end
save("GroupTHOIgS.mat","GroupTHOIgS");
save("GroupBinHOIgS.mat","GroupBinHOIgS");
save("GroupFDRHOIgS.mat","GroupFDRHOIgS");
save("GroupUncPHOIgS.mat","GroupUncPHOIgS");
save("GroupAvHOIgS.mat","GroupAvHOIgS")
