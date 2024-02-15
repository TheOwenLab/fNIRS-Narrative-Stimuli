%% compute group FDR correction
% at channel level
load("MaxT"); load("ISC.mat");

% drop rest condition 
ISC = ISC(1:4,:,:,:);

% set loop parameters
[nCond,nChan,nHb,nSub] = size(ISC);
nIter = size(MaxT,2);

% preallocate variables
GroupFDRBin = nan(nCond,nChan); % 0 (ns) 1 sig
GroupFDRSum = nan(nCond,nChan); % corrected p 
GroupFDRT = nan(nCond,nChan); % t-score
GroupUncP = nan(nCond,nChan); % uncorrected p
GroupAv = nan(nCond,nChan); % group averaged ISCs

% for each condition
for nc = 1:nCond
    % get null across subject for each condition
    maxt = mean(squeeze(MaxT(nc,:)),1);
    % get cutoff
    tstar = prctile(maxt,95)
    % actual values
    aISC = squeeze(mean(ISC(nc,:,:,:),3));

    for cc = 1:nChan
        % generate test statistic
        [~,GroupUncP(nc,cc),~,STATS] = ttest(aISC(cc,:),0,'tail','right');
        % store group level statistics
        GroupFDRT(nc,cc) =  STATS.tstat;
        GroupFDRBin(nc,cc) = STATS.tstat > tstar;
        GroupFDRSum(nc,cc) = sum((STATS.tstat < maxt)/nIter);
    end
    GroupAv(nc,:) = nanmean(aISC,2);

end
save("GroupFDRT.mat","GroupFDRT");
save("GroupFDRBin.mat","GroupFDRBin");
save("GroupFDRSum.mat","GroupFDRSum");
save("GroupUncP.mat","GroupUncP");
save("GroupAv.mat","GroupAv");

%% for IgS 
load("MaxTIgS.mat"); load("ISC.mat");
% drop rest
ISC = ISC(1:4,:,:,:);

% set loop parameters
[nCond,nChan,nHb,nSub] = size(ISC);
nIter = size(MaxTIgS,2);

% preallocate variables
GroupFDRBinIgS = nan(nCond/2,nChan);  % 0 (ns) 1 sig
GroupFDRSumIgS = nan(nCond/2,nChan); % corrected p 
GroupFDRTIgS = nan(nCond/2,nChan);  % tscore
GroupUncPIS = nan(nCond/2,nChan); % uncorrected P

% for each condition
counter = 0;
for nc = 1:2:nCond
   counter = counter + 1;
    % get null across subject for each condition
    maxt = squeeze(mean(MaxTIgS(counter,:,:),1));
    % get cutoff
    tstar = prctile(maxt,95)
    % actual values
    aISCi = squeeze(ISC(nc,:,:,:));
    aISCs = squeeze(ISC(nc+1,:,:,:));
    rdiff = squeeze((aISCi(:,1,:) - aISCs(:,1,:)) + (aISCi(:,2,:) - aISCs(:,2,:)));
    rdiff = rdiff/2;
    for cc = 1:nChan
        % generate test statistic
        [~,GroupUncPIS(nc,cc),~,STATS] = ttest(rdiff(cc,:),0,'tail','right');
        % store group level statistics
        GroupFDRTIgS(counter,cc) =  STATS.tstat;
        GroupFDRBinIgS(counter,cc) = STATS.tstat > tstar;
        GroupFDRSumIgS(counter,cc) = sum(STATS.tstat < maxt)/nIter;
    end
end
save("GroupFDRTIgS.mat","GroupFDRTIgS");
save("GroupFDRBinIgS.mat","GroupFDRBinIgS");
save("GroupFDRSumIgS.mat","GroupFDRSumIgS");
save("GroupUncPIS.mat","GroupUncPIS");
