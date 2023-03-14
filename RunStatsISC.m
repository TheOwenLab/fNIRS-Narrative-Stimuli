%% Conduct Permuatation Statistics

% load in data 
load("ISC.mat"); load("ISCNull.mat");

% drop rest data
ISC = ISC(1:4,:,:,:);

[nCond,nChan,nHb,nSub] = size(ISC);

ISCRes = nan(nCond,nChan,nSub);
nIter = size(ISCNull,4);

% for each condition
for nc = 1:nCond
    % for each subject
    for ss = 1:nSub         
        % for each channel 
        for cc = 1:nChan
            % find the max value from the null distribution
            mISC = (squeeze(ISCNull(nc,cc,ss,:)));
            % actual value
            aISC = mean(ISC(nc,cc,:,ss),3);
            % compute p-value
            ISCRes(nc,cc,ss) = sum(aISC < mISC)/nIter;              
        end
    end
end

save("PSub.mat","ISCRes");

%% generate t-score rather than p-value

% same for average hbo and hbr
ISCResAverageT = nan(nCond,nChan,nSub);
% for each condition
for nc = 1:nCond
    % for each subject
    for ss = 1:nSub        
        % for each channel 
        for cc = 1:nChan
            % find the max value from the null distribution
            mISC = squeeze(ISCNull(nc,cc,ss,:));
            % actual value
            aISC = squeeze(mean(ISC(nc,cc,:,ss),3));
            % compute t-value
            mu = mean(mISC); sigma = std(mISC);
            ISCResAverageT(nc,cc,ss) = (aISC - mu)/sigma;              

        end
    end
end
save("TSub.mat","ISCResAverageT");

%% compare intact vs scrambled at the participant level?
ISCResAverageT = nan(nCond/2,nChan,nSub);
counter = 0;
% for each condition
for nc = 1:2:nCond
    counter = counter + 1; 
    % for each subject
    for ss = 1:nSub        
        % for each channel 
        for cc = 1:nChan
            % find the max value from the relevent null distributions
            mISCI = squeeze(ISCNull(nc,cc,ss,:)); mISCS = squeeze(ISCNull(nc+1,cc,ss,:));
            % actual values
            aISCI = squeeze(mean(ISC(nc,cc,:,ss),3)); aISCS = squeeze(mean(ISC(nc+1,cc,:,ss),3));
            % get parameters from null distribution
            mui = mean(mISCI); sigmai = std(mISCI);
            mus = mean(mISCS); sigmas = std(mISCS);
            % compute z-value ((x1 - x2) - (u1 - u2))/(s1 + s2)
            z = ((aISCI - aISCS) - (mui - mus))/((sigmai + sigmas)/2);
            ISCResAverageT(counter,cc,ss) = z;              

        end
    end
end
save("TSubIS.mat","ISCResAverageT");


%% Mean DIFFERENCE Group level Average (Within Sub)

ISCResAverageDiffG = nan(nCond,nChan);
GroupP = nan(nCond,nChan); GroupFDR = nan(nCond,nChan);
% for each condition
for nc = 1:nCond       
    % for each channel 
    for cc = 1:nChan
        % find the max value from the null distribution
        %mISC = (squeeze(mean(ISCNull(nc,cc,:,:,:),3)));
        mISC = squeeze(ISCNull(nc,cc,:,:));
        mu = mean(mISC,2); % average over subjects
        sigma = std(mISC,[],2); %std over subjects
        % actual value
        aISC = squeeze(mean(ISC(nc,cc,:,:),3));
        % compute t-value
        gISC = (aISC - mu)./sigma;
        [H,P,CI,ZVAL] = ttest(gISC,0,'tail','right'); % just a ttest
        ISCResAverageDiffG(nc,cc) = ZVAL.tstat;
        % store group level p-vals
        GroupP(nc,cc) = P;
        

    end
    [~,~,~,GroupFDR(nc,:)] = fdr_bh(GroupP(nc,:));
end
save("TGroup.mat","ISCResAverageDiffG");
save("GroupP.mat","GroupP");
save("GroupFDR.mat","GroupFDR");
%% determine group level I > S
IS = nan(nCond/2,nChan);
ISP = nan(nCond/2,nChan); ISPFDR = nan(nCond/2,nChan);
% for each condition
counter = 1;
for nc = 1:2:nCond       
    % for each channel 
    for cc = 1:nChan
        % find the max value from the null distribution
        mNI = squeeze(ISCNull(nc,cc,:,:)); mNS = squeeze(ISCNull(nc+1,cc,:,:));
        muI = mean(mNI,2); muS = mean(mNS,2); % average over subjects
        sigmaI = std(mNI,[],2); sigmaS = std(mNS,[],2); %std over subjects
        % actual ISC values
        I = squeeze(mean(ISC(nc,cc,:,:),3)); S = squeeze(mean(ISC(nc+1,cc,:,:),3));
        % compute z-values
        gI = (I - muI)./sigmaI; gS = (S - muS)./sigmaS;
        [H,P,CI,ZVAL] = ttest(gI,gS,'tail','right');
        IS(counter,cc) = ZVAL.tstat;
        % store group level p-vals
        ISP(counter,cc) = P;    
    end
    [~,~,~,ISPFDR(counter,:)] = fdr_bh(ISP(counter,:));
    counter = counter + 1;
end
save("IS.mat","IS");
save("ISP.mat","ISP");
save("ISPFDR.mat","ISPFDR");


%% determine group level S > I
SI = nan(nCond/2,nChan);
SIP = nan(nCond/2,nChan); SIPFDR = nan(nCond/2,nChan);
% for each condition
counter = 1;
for nc = 1:2:nCond       
    % for each channel 
    for cc = 1:nChan
        % find the max value from the null distribution
        %mISC = (squeeze(mean(ISCNull(nc,cc,:,:,:),3)));
        mNI = squeeze(ISCNull(nc,cc,:,:)); mNS = squeeze(ISCNull(nc+1,cc,:,:));
        muI = mean(mNI,2); muS = mean(mNS,2); % average over subjects
        sigmaI = std(mNI,[],2); sigmaS = std(mNS,[],2); %std over subjects
        % actual ISC values
        I = squeeze(mean(ISC(nc,cc,:,:),3)); S = squeeze(mean(ISC(nc+1,cc,:,:),3));
        % compute z-values
        gI = (I - muI)./sigmaI; gS = (S - muS)./sigmaS;
        [H,P,CI,ZVAL] = ttest(gS,gI,'tail','right');
        SI(counter,cc) = ZVAL.tstat;
        % store group level p-vals
        SIP(counter,cc) = P;
    end
    [~,~,~,SIPFDR(counter,:)] = fdr_bh(SIP(counter,:));
    counter = counter + 1;
end
save("SI.mat","SI");
save("SIP.mat","SIP");
save("SIPFDR.mat","SIPFDR");


%% determine REAL group level I > S
IR = nan(nCond/4,nChan);
IRP = nan(nCond/4,nChan); IRPFDR = nan(nCond/4,nChan);
% for each condition
counter = 1; ICond = [1,3]; SCond = [2,4];    
% for each channel 
for cc = 1:nChan
    % find the max value from the null distribution
    %mISC = (squeeze(mean(ISCNull(nc,cc,:,:,:),3)));
    mNIB = squeeze(ISCNull(ICond(1),cc,:,:)); mNIT = squeeze(ISCNull(ICond(2),cc,:,:));  
    mNSBS = squeeze(ISCNull(SCond(1),cc,:,:)); mNSTS = squeeze(ISCNull(SCond(2),cc,:,:));
    muIB = mean(mNIB,2); muIT = mean(mNIT,2); 
    muSBS = mean(mNSBS,2); muSTS = mean(mNSTS,2);% average over subjects
    sigmaIB = std(mNIB,[],2); sigmaIT = std(mNIT,[],2); 
    sigmaSBS = std(mNSBS,[],2); sigmaSTS = std(mNSTS,[],2); %std over subjects
    % actual ISC values
    IB = squeeze(mean(ISC(ICond(1),cc,:,:),3)); IT = squeeze(mean(ISC(ICond(2),cc,:,:),3)); 
    SB = squeeze(mean(ISC(SCond(1),cc,:,:),3)); ST = squeeze(mean(ISC(SCond(2),cc,:,:),3));

    % compute z-values
    gIB = (IB - muIB)./sigmaIB; gIT = (IT - muIT)./sigmaIT; 
    gSB = (SB - muSBS)./sigmaSBS; gST = (ST - muSTS)./sigmaSTS;
    % then average these values
    gI = (gIB + gIT)/2; gS = (gSB + gST)/2;
    [H,P,CI,ZVAL] = ttest(gI,gS,'tail','right');
    IR(counter,cc) = ZVAL.tstat;
    % store group level p-vals
    IRP(1,cc) = P;

end
[~,~,~,IRPFDR(1,:)] = fdr_bh(IRP(1,:));

save("ISrRes.mat","IR");
save("ISrP.mat","IRP");
save("ISrPFDR.mat","IRPFDR");



%% determine REAL group level S > I
SR = nan(nCond/4,nChan);
SRP = nan(nCond/4,nChan); SRPFDR = nan(nCond/4,nChan);
% for each condition
counter = 1; ICond = [1,3]; SCond = [2,4];    
% for each channel 
for cc = 1:nChan
    % find the max value from the null distribution
    %mISC = (squeeze(mean(ISCNull(nc,cc,:,:,:),3)));
    mNIB = squeeze(ISCNull(ICond(1),cc,:,:)); mNIT = squeeze(ISCNull(ICond(2),cc,:,:));  
    mNSBS = squeeze(ISCNull(SCond(1),cc,:,:)); mNSTS = squeeze(ISCNull(SCond(2),cc,:,:));
    muIB = mean(mNIB,2); muIT = mean(mNIT,2); 
    muSBS = mean(mNSBS,2); muSTS = mean(mNSTS,2);% average over subjects
    sigmaIB = std(mNIB,[],2); sigmaIT = std(mNIT,[],2); 
    sigmaSBS = std(mNSBS,[],2); sigmaSTS = std(mNSTS,[],2); %std over subjects
    % actual ISC values
    IB = squeeze(mean(ISC(ICond(1),cc,:,:),3)); IT = squeeze(mean(ISC(ICond(2),cc,:,:),3)); 
    SB = squeeze(mean(ISC(SCond(1),cc,:,:),3)); ST = squeeze(mean(ISC(SCond(2),cc,:,:),3));

    % compute z-values
    gIB = (IB - muIB)./sigmaIB; gIT = (IT - muIT)./sigmaIT; 
    gSB = (SB - muSBS)./sigmaSBS; gST = (ST - muSTS)./sigmaSTS;
    % then average these values
    gI = (gIB + gIT)/2; gS = (gSB + gST)/2;
    [H,P,CI,ZVAL] = ttest(gS,gI,'tail','right');
    SR(counter,cc) = ZVAL.tstat;
    % store group level p-vals
    SRP(1,cc) = P;


end
[~,~,~,SRPFDR(1,:)] = fdr_bh(SRP(1,:));

save("SIrRes.mat","SR");
save("SIrP.mat","SRP");
save("SIrPFDR.mat","SRPFDR");


%% determine group level I > S using ISNull
load("ISCNullIS.mat")
IS = nan(nCond/2,nChan);
ISP = nan(nCond/2,nChan); ISPFDR = nan(nCond/2,nChan);
% for each condition
counter = 1;
for nc = 1:2:nCond       
    % for each channel 
    for cc = 1:nChan
        % collect the values from the null distribution
        mN = squeeze(ISCNull(counter,cc,:,:));
        mu = mean(mN,2);
        sigma = std(mN,[],2);
        % actual ISC values
        I = squeeze(mean(ISC(nc,cc,:,:),3)); S = squeeze(mean(ISC(nc+1,cc,:,:),3));
        % compute z-values
        g = I - S;
        gZ = (g-mu)./sigma;
        [H,P,CI,ZVAL] = ttest(gZ,0,'tail','both');
        IS(counter,cc) = ZVAL.tstat;
        % store group level p-vals
        ISP(counter,cc) = P;    
    end
    [~,~,~,ISPFDR(counter,:)] = fdr_bh(ISP(counter,:));
    counter = counter + 1;
end
save("ISp.mat","IS");
save("ISPp.mat","ISP");
save("ISPFDRp.mat","ISPFDR");
