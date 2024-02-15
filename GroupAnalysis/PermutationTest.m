%% Permutation Testing to compare Taken, BYD, BYD Scrambled and Taken Scrambled to chance 
% (operationalized as phase scrambled surrogates of the original data. 
% See Prichard, D., & Theiler, J. (1994). Generating surrogate data for time series with several simultaneously measured variables. Physical Review Letters, 73(7), 951–954. https://doi.org/10.1103/PhysRevLett.73.951
% Groups statistics of the ISCs calculated from phase scrambled surrogates 
% are corrected via the max-t method (Nichols and Holmes, 2002)
% Nichols, T. E., & Holmes, A. P. (2002). Nonparametric permutation tests for functional neuroimaging: A primer with examples. Human Brain Mapping, 15(1), 1–25. https://doi.org/10.1002/hbm.1058

% load in data
filename = "PreprocessedData.mat";
data = load(filename);
data = data.data;

% set loop parameters
nSub = length(data); nCond = 4; 
nChan = 121; nHb = 2; nIter = 1000;
SSlist = [8 29 52 66 75 92 112 125];

% put data in easy to use format
BYD = cat(4,data.BYD); BYDN = cat(4,data.BYDN);
Taken = cat(4,data.Taken); TakenNoise = cat(4,data.TakenNoise);

% drop to odd values for the fft
BYD = BYD(1:end-1,:,:,:);
Group = {BYD;BYDN;Taken;TakenNoise};

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
ISCNull = nan(nCond,nChan,nSub); % for a given permutation
MaxT = nan(nCond,nIter); % to collect max-ts 

rng(59); % set random seed

for nn = 1:nIter
   % get subject and subject - 1 data for each condition
   nn
    for nc=1:nCond
        % for each subject
        for ss=1:nSub
            pRand = rand(length(Group{nc}),nChan); % set randomization
            phi = (pRand .* 2*pi); 
            % split subject and mean group
            gs = setdiff(1:nSub,ss);
            % grab condition data for subject and left out subjects
            scdata = Group{nc}(:,:,:,ss); gcdata = Group{nc}(:,:,:,gs);
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
                BC = BadChannels(cc,gs);
                cleangroup = gcdata(:,cc,:,~BC);
                cleangroup = squeeze(mean(cleangroup,4));
                oISC = atanh(corr(sosig(:,cc),cleangroup(:,1))); rISC = atanh(corr(srsig(:,cc),cleangroup(:,2))); 
            % then average over hbs and store result               
                tempISC = [tempISC (oISC + rISC)/2];
            end
            ISCNull(nc,:,ss) = tempISC;

        end
        % run group level statistical test to obtain t-scores that are
        % obtain by chance
        tmax = [];
        for cc = 1:nChan
            r = squeeze(ISCNull(nc,cc,:));
            [~,~,~,STATS] = ttest(r,0,'tail','right');
            tmax = [tmax,STATS.tstat];
        end  
        MaxT(nc,nn) = max(tmax);
        
    end
end


toc 
% save results
save("MaxT.mat","MaxT")
%% see the results


% hists across subjects and conditions
figure
histogram(squeeze(MaxT(1,:)));

figure
histogram(squeeze(MaxT(2,:)));

figure
histogram(squeeze(MaxT(3,:)));

figure
histogram(squeeze(MaxT(4,:)));



