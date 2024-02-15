%% run permutation testing to calcualte ISC for phase scrambled surrogates for hold out datasets

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

% preallocate ISC results
nIter = 1000;
ISCNull = nan(nCond,nChan,nSub-1);
MaxTHoldOut = nan(nCond,nIter);
rng(59);
randHO = randi(nSub,[nIter,1]);
subList = 1:nSub;

%% run permutation testing 
tic
for nn = 1:nIter
    % work on a random hold out set to deal with computational issues
    hoSubList = subList~=randHO(nn);
    tGroup = {Group{1}(:,:,:,hoSubList),Group{2}(:,:,:,hoSubList),...
        Group{3}(:,:,:,hoSubList),Group{4}(:,:,:,hoSubList)};
    for nc=1:nCond
        % for each subject
        parfor ss=1:nSub-1
            pRand = rand(length(tGroup{nc}),nChan);
            phi = (pRand .* 2*pi); 
            % split subject from group
            gs = setdiff(1:nSub-1,ss);
            % grab condition data for subject and left out subjects
            scdata = tGroup{nc}(:,:,:,ss); gcdata = tGroup{nc}(:,:,:,gs);
            % get channel data for left out and average group  
            % grab hbo
            scotemp = squeeze(scdata(:,:,1)); 
            % grab hbr
            scrtemp = squeeze(scdata(:,:,2)); 
            %theta = angle(z) returns the phase angle in the interval [-π,π] for each element of a complex array z. 
            %The angles in theta are such that z = abs(z).*exp(i*theta).
            fo = fft(scotemp); fr = fft(scrtemp);
            % A "phase-randomized" Fourier transform X(f) is made by rotating the phase P at 
            % each frequency f by an independent random variable 
            % P which is chosen uniformly in the range [0, 2pi).
            % reconstruct timeseries with new randomized phase
            % and from this the surrogate time series is given by the inverse Fourier transform:
            sosig = real(ifft(fo .* exp(1i*phi))); srsig = real(ifft(fr .* exp(1i*phi)));
            % fisher transformed version of the correlation between group
            tempISC = [];
            for cc = 1:nChan
                % drop poor channels from group
                BC = BadChannels(:,hoSubList);
                BC = BC(cc,gs);
                cleangroup = gcdata(:,cc,:,~BC);
                cleangroup = squeeze(mean(cleangroup,4));
                oISC = atanh(corr(sosig(:,cc),cleangroup(:,1))); rISC = atanh(corr(srsig(:,cc),cleangroup(:,2))); 
            % then average over hbs and store result               
                tempISC = [tempISC (oISC + rISC)/2];
            end
            ISCNull(nc,:,ss) = tempISC;

        end
        % run group level statistical test
        tmax = [];
        for cc = 1:nChan
            r = squeeze(ISCNull(nc,cc,:));
            [~,~,~,STATS] = ttest(r,0,'tail','right');
            tmax = [tmax,STATS.tstat];
        end  
        MaxTHoldOut(nc,nn) = max(tmax);
        
    end
end

toc 
% save results
save("MaxTHoldOut.mat","MaxTHoldOut")

%% plot the distribution of hold outs
% hists across subjects and conditions
figure
histogram(squeeze(MaxTHoldOut(1,:)));

figure
histogram(squeeze(MaxTHoldOut(2,:)));

figure
histogram(squeeze(MaxTHoldOut(3,:)));

figure
histogram(squeeze(MaxTHoldOut(4,:)));



