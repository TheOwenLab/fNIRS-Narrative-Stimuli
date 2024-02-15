%% ISC permutation stats

% Each voxel's time course was phase-scrambled by taking the Fast Fourier Transform of the signal, 
% randomizing the phase of each Fourier component, and then inverting the Fourier transformation. 
% This randomization procedure thus only scrambles the phase of the signal, leaving its power spectrum intact. 
% Using the phase-scrambled surrogate dataset, the ISC was again calculated for all voxels as described above, 
% creating a null distribution of average correlation values for each voxel. 
% This bootstrapping procedure was repeated 1000 times, producing 1000 bootstrapped correlation maps (Regev et al., 2013; Honey et al., 2012).

% from Regev & Honey Familywise error rate (FWER) was defined as the top 5% 
% of the null distribution of the maximum correlations values exceeding a given threshold (R*), 



%% ISC 

% load in data
filename = "PreprocessedData.mat";
data = load(filename);
data = data.data;
% set loop parameters
nSub = length(data); nCond = 4; 
nChan = 121; nHb = 2;

% put data in easy to use format
BYD = cat(4,data.BYD); BYDN = cat(4,data.BYDN);
Taken = cat(4,data.Taken); TakenNoise = cat(4,data.TakenNoise);
Group = {BYD;BYDN;Taken;TakenNoise};
SSlist = [8 29 52 66 75 92 112 125];

% collect bad channels
load("PreprocessedDataCWNIRS.mat")
BadChannels = zeros(nChan,nSub);

% for each participant
for ss = 1:nSub
    % move from sc+lc space to lc space
    temp = nan(129,1);
    temp(gdata{ss}.SD.BadChannels) = 1;
    temp(SSlist) = [];
    BadChannels(temp == 1,ss) = 1;
end

clear gdata


%% run permutation statistics for average ISCs

% preallocate ISC results
nIter = 1000;
ISCNullIS= nan(nCond/2,nChan,nSub-1);
MaxTHoldOutIS = nan(nCond/2,nIter);
counter = 0;
rng(59);
randHO = randi(nSub,[nIter,1]);
subList = 1:nSub;
tic
for nn = 1:nIter
    nn
    % work on random hold out set
    hoSubList = subList~=randHO(nn);
    tGroup = {Group{1}(:,:,:,hoSubList),Group{2}(:,:,:,hoSubList),...
        Group{3}(:,:,:,hoSubList),Group{4}(:,:,:,hoSubList)};
   % get subject and subject - 1 data for each condition
    counter = 0;
    for nc=1:2:nCond
        counter = counter + 1;
        % for each subject
        parfor ss=1:nSub-1
            % set randomization vector for phase
            pRandi = rand(length(tGroup{nc}),nChan);
            phii = (pRandi .* 2*pi); 
            pRands = rand(length(tGroup{nc+1}),nChan);
            phis = (pRands .* 2*pi);
            % split subject and mean group
            gs = setdiff(1:nSub-1,ss);
            % grab condition data for subject and left out subjects
            scidata = tGroup{nc}(:,:,:,ss); gcidata = tGroup{nc}(:,:,:,gs); 
            scsdata = tGroup{nc+1}(:,:,:,ss); gcsdata = tGroup{nc+1}(:,:,:,gs); 
            
            % get channel data for left out and average group 
            % first for intact
            % grab hbo
            scoitemp = squeeze(scidata(:,:,1)); gcoitemp = squeeze(gcidata(:,:,1,:));
            % grab hbr
            scritemp = squeeze(scidata(:,:,2)); gcritemp = squeeze(gcidata(:,:,2,:));
            % then for scrambled
            % grab hbo
            scostemp = squeeze(scsdata(:,:,1)); gcostemp = squeeze(gcsdata(:,:,1,:));
            % grab hbr
            scrstemp = squeeze(scsdata(:,:,2)); gcrstemp = squeeze(gcsdata(:,:,2,:));
            %theta = angle(z) returns the phase angle in the interval [-π,π] for each element of a complex array z. 
            %The angles in theta are such that z = abs(z).*exp(i*theta).
            foi = fft(scoitemp); fri = fft(scritemp);
            fos = fft(scostemp); frs = fft(scrstemp);
            % A "phase-randomized" Fourier transform X(f) is made by rotating the phase P at 
            % each frequency f by an independent random variable 
            % P which is chosen uniformly in the range [0, 2pi).
            % reconstruct timeseries with new randomized phase
            % and from this the surrogate time series is given by the inverse Fourier transform:
            soisig = real(ifft(foi .* exp(1i*phii))); srisig = real(ifft(fri .* exp(1i*phii)));
            sossig = real(ifft(fos .* exp(1i*phis))); srssig = real(ifft(frs .* exp(1i*phis)));
            % fisher transformed version of the correlation between group
            tempISC = [];
            for cc = 1:nChan
                 % drop poor channels from group
                % drop poor channels from group
                BC = BadChannels(:,hoSubList);
                BC = BC(cc,gs);
                % first for intact
                % hbo
                cleangroupio = gcoitemp(:,cc,~BC);
                cleangroupio = squeeze(mean(cleangroupio,3));
                % hbr
                cleangroupir = gcritemp(:,cc,~BC);
                cleangroupir = squeeze(mean(cleangroupir,3));
                
                % scrambled
                % hbo
                cleangroupso = gcostemp(:,cc,~BC);
                cleangroupso = squeeze(mean(cleangroupso,3));               
                % hbr 
                cleangroupsr = gcrstemp(:,cc,~BC);
                cleangroupsr = squeeze(mean(cleangroupsr,3));
                
                % compute intact and scrambled statistics
                oISCi = atanh(corr(soisig(:,cc),cleangroupio)); rISCi = atanh(corr(srisig(:,cc),cleangroupir)); 
                oISCs = atanh(corr(sossig(:,cc),cleangroupso)); rISCs = atanh(corr(srssig(:,cc),cleangroupsr));
                stat = (oISCi - oISCs) + (rISCi - rISCs);

            % then average over hbs and store result               
                tempISC = [tempISC stat/2];
            end
            ISCNullIS(counter,:,ss) = tempISC;

        end
        % run group level statistical test
        tmax = [];
        for cc = 1:nChan
            r = squeeze(ISCNullIS(counter,cc,:));
            [~,~,~,STATS] = ttest(r,0,'tail','right');
            tmax = [tmax,STATS.tstat];
        end 
        % both ends of the distribution
        MaxTHoldOutIS(counter,nn) = max(tmax);
        
    end
end


toc 
% save results
save("MaxTHoldOutIS.mat","MaxTHoldOutIS")
%% see the results


% hists across subjects and conditions
figure
histogram(squeeze(MaxTHoldOutIS(1,:)));

figure
histogram(squeeze(MaxTHoldOutIS(2,:)));


