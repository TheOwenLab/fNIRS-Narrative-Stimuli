%% Conduct Permuatation Statistics

% load in data 
load("ISCHoldOut.mat"); load("ISCNullHoldOut.mat");

[nCond,nChan,nSub,nSubList] = size(ISC);

ISCRes = nan(nCond,nChan,nSub,nSubList);
nIter = size(ISCNull,5);

% for each condition
for nc = 1:nCond
    % for each left out subject
    for ss = 1:nSub
        % for each remaining subjects .
        for rs = 1:nSubList
            % for each channel 
            for cc = 1:nChan
                % find the values from the null distribution
                mISC = (squeeze(ISCNull(nc,cc,ss,rs,:)));
                % actual value
                aISC = mean(ISC(nc,cc,ss,rs),3);
                % compute p-value
                ISCRes(nc,cc,ss,rs) = sum(aISC < mISC)/nIter;              
            end
        end
    end
end

save("PSubML.mat","ISCRes");

%% generate t-score rather than p-value

% same for average hbo and hbr
ISCResAverageT = nan(nCond,nChan,nSub,nSubList);
% for each condition
for nc = 1:nCond
    % for each subject
    for ss = 1:nSub
        % for each remaining subjects .
        for rs = 1:nSubList
            % for each channel 
            for cc = 1:nChan
                % find the max value from the null distribution
                mISC = squeeze(ISCNull(nc,cc,ss,rs,:));
                % actual value
                aISC = ISC(nc,cc,ss,rs);
                % compute t-value
                mu = mean(mISC); sigma = std(mISC);
                ISCResAverageT(nc,cc,ss,rs) = (aISC - mu)/sigma;    
            end

        end
    end
end
save("TSubML.mat","ISCResAverageT");

%% Compute FDR within each left out dataset
ISCResAverageDiffG = nan(nCond,nChan,nSub);
GroupP = nan(nCond,nChan,nSub); GroupFDR = nan(nCond,nChan,nSub);
% for each condition
for nc = 1:nCond
    % for each left out subject.
    for ss = 1:nSub
        % for each channel 
        for cc = 1:nChan
            % find the max value from the null distribution
            %mISC = (squeeze(mean(ISCNull(nc,cc,:,:,:),3)));
            mISC = squeeze(ISCNull(nc,cc,ss,:,:));
            mu = mean(mISC,2); % average over subjects
            sigma = std(mISC,[],2); %std over subjects
            % actual value
            aISC = squeeze(ISC(nc,cc,ss,:));
            % compute t-value
            gISC = (aISC - mu)./sigma;
            [H,P,CI,ZVAL] = ttest(gISC,0,'tail','right'); % just a ttest
            ISCResAverageDiffG(nc,cc,ss) = ZVAL.tstat;
            % store group level p-vals
            GroupP(nc,cc,ss) = P;

        end
     [~,~,~,GroupFDR(nc,:,ss)] = fdr_bh(GroupP(nc,:,ss));
    end
   
end
save("TGroupML.mat","ISCResAverageDiffG");
save("GroupPML.mat","GroupP");
save("GroupFDRML.mat","GroupFDR");

%% Compute FDR within each left out dataset
ISCResAverageDiffG = nan(nCond,nChan,nSub);
GroupP = nan(nCond,nChan,nSub); GroupFDR = nan(nCond,nChan,nSub);
% for each condition
for nc = 1:nCond
    % for each left out subject.
    for ss = 1:nSub
        % for each channel 
        for cc = 1:nChan
            
            mISC = squeeze(ISCNull(nc,cc,ss,:,:));
            mu = mean(mISC,2); % average over subjects
            sigma = std(mISC,[],2); %std over subjects
            % actual value
            aISC = squeeze(ISC(nc,cc,ss,:));
            % compute t-value
            gISC = (aISC - mu)./sigma;
            [H,P,CI,ZVAL] = ttest(gISC,0,'tail','right'); % just a ttest
            ISCResAverageDiffG(nc,cc,ss) = ZVAL.tstat;
            % store group level p-vals
            GroupP(nc,cc,ss) = P;

        end
     [~,~,~,GroupFDR(nc,:,ss)] = fdr_bh(GroupP(nc,:,ss));
    end
   
end
save("TGroupML.mat","ISCResAverageDiffG");
save("GroupPML.mat","GroupP");
save("GroupFDRML.mat","GroupFDR");

%% determine group level I > S
IS = nan(nCond/2,nChan,nSub);
ISP = nan(nCond/2,nChan,nSub); ISPFDR = nan(nCond/2,nChan,nSub);
% for each condition
counter = 1;
for nc = 1:2:nCond 
    % for each left out subject.
    for ss = 1:nSub
        % for each channel 
        for cc = 1:nChan
            % find the max value from the null distribution
            mNI = squeeze(ISCNull(nc,cc,ss,:,:)); mNS = squeeze(ISCNull(nc+1,cc,ss,:,:));
            muI = mean(mNI,2); muS = mean(mNS,2); % average over subjects
            sigmaI = std(mNI,[],2); sigmaS = std(mNS,[],2); %std over subjects
            % actual ISC values
            I = squeeze(ISC(nc,cc,ss,:)); S = squeeze(ISC(nc+1,cc,ss,:));
            % compute z-values
            gI = (I - muI)./sigmaI; gS = (S - muS)./sigmaS;
            [H,P,CI,ZVAL] = ttest(gI,gS,'tail','right');
            IS(counter,cc,ss) = ZVAL.tstat;
            % store group level p-vals
            ISP(counter,cc,ss) = P;    
        end
    [~,~,~,ISPFDR(counter,:,ss)] = fdr_bh(ISP(counter,:,ss));
    
    end
    counter = counter + 1;
end
save("ISML.mat","IS");
save("ISPML.mat","ISP");
save("ISPFDRML.mat","ISPFDR");



%% determine condition null
nIterList = 1:nIter;
NullRes = nan(nCond,nChan,nSub,nIter);
NullResP = nan(nCond,nChan,nSub,nIter); NullResFDR = nan(nCond,nChan,nSub,nIter);
% for each condition
for nn = 1:nIter
    counter = 1;
    % first, remove iter from list 
    tsList = nIterList(nIterList ~= nn);
    for nc = 1:nCond 
        % for each left out subject.
        for ss = 1:nSub
            % for each channel 
            for cc = 1:nChan
                % find the max value from the null distribution
                mISC = squeeze(ISCNull(nc,cc,ss,:,tsList));
                mu = mean(mISC,2); % average over subjects
                sigma = std(mISC,[],2); %std over subjects
                % chosen value from null distribution
                aISC = squeeze(ISCNull(nc,cc,ss,:,nn)); 
                % compute z-values
                gISC = (I - muI)./sigmaI; 
                [H,P,CI,ZVAL] = ttest(gISC,0,'tail','right');
                NullRes(counter,cc,ss,nn) = ZVAL.tstat;
                % store group level p-vals
                NullResP(counter,cc,ss,nn) = P;    
            end
        [~,~,~,NullResFDR(counter,:,ss,nn)] = fdr_bh(NullResP(counter,:,ss,nn));

        end
        counter = counter + 1;
    end
end
save("TGroupMLNULL.mat","NullRes");
save("TGroupPMLNULL.mat","NullResP");
save("TGroupPFDRMLNULL.mat","NullResFDR");
%% determine IS null

%determine group level I > S
nIterList = 1:nIter;
IS = nan(nCond/2,nChan,nSub,nIter);
ISP = nan(nCond/2,nChan,nSub,nIter); ISPFDR = nan(nCond/2,nChan,nSub,nIter);
% for each condition
for nn = 1:nIter
    counter = 1;
    % first, remove iter from list 
    tsList = nIterList(nIterList ~= nn);
    for nc = 1:2:nCond 
        % for each left out subject.
        for ss = 1:nSub
            % for each channel 
            for cc = 1:nChan
                % find the max value from the null distribution
                mNI = squeeze(ISCNull(nc,cc,ss,:,tsList)); mNS = squeeze(ISCNull(nc+1,cc,ss,:,tsList));
                muI = mean(mNI,2); muS = mean(mNS,2); % average over subjects
                sigmaI = std(mNI,[],2); sigmaS = std(mNS,[],2); %std over subjects
                % chosen value from null distribution
                I = squeeze(ISCNull(nc,cc,ss,:,nn)); S = squeeze(ISCNull(nc+1,cc,ss,:,nn));
                % compute z-values
                gI = (I - muI)./sigmaI; gS = (S - muS)./sigmaS;
                [H,P,CI,ZVAL] = ttest(gI,gS,'tail','right');
                IS(counter,cc,ss,nn) = ZVAL.tstat;
                % store group level p-vals
                ISP(counter,cc,ss,nn) = P;    
            end
        [~,~,~,ISPFDR(counter,:,ss,nn)] = fdr_bh(ISP(counter,:,ss,nn));

        end
        counter = counter + 1;
    end
end
save("ISMLNULL.mat","IS");
save("ISPMLNULL.mat","ISP");
save("ISPFDRMLNULL.mat","ISPFDR");

%% determine group level I > S using ISNull
load("ISCNullHoldOutIS.mat")
IS = nan(nCond/2,nChan,nSub);
ISP = nan(nCond/2,nChan,nSub); ISPFDR = nan(nCond/2,nChan,nSub);
% for each condition
counter = 1;
for nc = 1:2:nCond 
    % for each left out subject.
    for ss = 1:nSub
        % for each channel 
        for cc = 1:nChan
            % find the max value from the null distribution
            mN = squeeze(ISCNull(counter,cc,ss,:,:));
            mu = mean(mN,2);
            sigma = std(mN,[],2); 
            % actual ISC values
            I = squeeze(ISC(nc,cc,ss,:)); S = squeeze(ISC(nc+1,cc,ss,:));
            % compute z-values
            g = I-S;
            gZ = (g - mu)./sigma;
            [H,P,CI,ZVAL] = ttest(gZ,0,'tail','both');
            IS(counter,cc,ss) = ZVAL.tstat;
            % store group level p-vals
            ISP(counter,cc,ss) = P;    
        end
    [~,~,~,ISPFDR(counter,:,ss)] = fdr_bh(ISP(counter,:,ss));
    
    end
    counter = counter + 1;
end
save("ISMLp.mat","IS");
save("ISPMLp.mat","ISP");
save("ISPFDRMLp.mat","ISPFDR");


%% for each iteration of null distribution count the number of significant channels
% start with BYD
load("ISPFDRMLNULL.mat")
BYDNull = squeeze(ISPFDR(1,:,:,:));
BYDsNull = squeeze(sum(BYDNull < 0.05,2));
max(BYDsNull)

% with lack of correction
BYDNullnc = squeeze(ISP(1,:,:,:));
BYDsNullnc = squeeze(sum(BYDNullnc < 0.05,2));
max(BYDsNullnc);

% then Taken
TakenNull = squeeze(ISPFDR(2,:,:,:));
TakensNull = squeeze(sum(TakenNull < 0.05,2));
max(TakensNull);

% with lack of correction
TakenNullnc = squeeze(ISP(2,:,:,:));
TakensNullnc = squeeze(sum(TakenNullnc < 0.05,2));
max(TakensNullnc);




