%% ISC permutation stats

% Each voxel's time course was phase-scrambled by taking the Fast Fourier Transform of the signal, 
% randomizing the phase of each Fourier component, and then inverting the Fourier transformation. 
% This randomization procedure thus only scrambles the phase of the signal, leaving its power spectrum intact. 
% Using the phase-scrambled surrogate dataset, the ISC was again calculated for all voxels as described above, 
% creating a null distribution of average correlation values for each voxel. 
% This bootstrapping procedure was repeated 1000 times, producing 1000 bootstrapped correlation maps (Regev et al., 2013; Honey et al., 2012).

% from Regev & Honey Familywise error rate (FWER) was defined as the top 5% 
% of the null distribution of the maximum correlations values exceeding a given threshold (R*), 



% load in data
filename = "PreprocessedData.mat";
data = load(filename);
data = data.data;
% set loop parameters
nSub = length(data); nCond = 4; 
nChan = 121; nHb = 2; nIter = 1000;

% put data in easy to use format
Rest = cat(4,data.Rest); BYD = cat(4,data.BYD); BYDN = cat(4,data.BYDN);
Taken = cat(4,data.Taken); TakenNoise = cat(4,data.TakenNoise);
% drop to odd values for the fft
Rest = Rest(1:end-1,:,:,:); BYD = BYD(1:end-1,:,:,:);
Group = {BYD;BYDN;Taken;TakenNoise};


% if computing autocorrelation
%l = 20; acPerm = nan(nCond,nChan,nHb,nSub,nIter,l); acNorm = nan(nCond,nChan,nHb,nSub,nIter,l);
%% run permutation statistics for average ISCs

% preallocate ISC results
ISCNull = nan(nCond,nChan,nSub,nIter);
tic
rng(59);
% for each iteration
for nn = 1:nIter
    % for each subject
    for ss=1:nSub
        % split subject and mean group
        gs = setdiff(1:nSub,ss);
        % get subject and subject - 1 data for each condition
        for nc=1:nCond
            scdata = Group{nc}(:,:,:,ss); gcdata = Group{nc}(:,:,:,gs);
            % finally, for each Hb type
            for cc = 1:nChan
                % set randomization
                %pRand = rand([floor((length(scdata)-1)/2), 1]);
                %gRand = rand([floor((length(scdata)-1)/2), 1]);
                pRand = rand(length(scdata),1);

                % get channel data for left out and group 
                sctemp = squeeze(scdata(:,cc,:)); gctemp = squeeze(gcdata(:,cc,:,:));
                % grab hbo
                scotemp = sctemp(:,1); gcotemp = mean(squeeze(gctemp(:,1,:)),2);
                % grab hbr
                scrtemp = sctemp(:,2); gcrtemp = mean(squeeze(gctemp(:,2,:)),2);
                % return phase randomzied signal for both hbs 
                %sosig = phaseranhb(scotemp,pRand); %gosig = phaseranhb(gcotemp,gRand);
                %srsig = phaseranhb(scrtemp,pRand); %grsig = phaseranhb(gcrtemp,gRand);
                
                %theta = angle(z) returns the phase angle in the interval [-π,π] for each element of a complex array z. 
                %The angles in theta are such that z = abs(z).*exp(i*theta).
                fo = fft(scotemp); fr = fft(scrtemp);
                % A "phase-randomized" Fourier transform X(f) is made by rotating the phase P at 
                % each frequency f by an independent random variable 
                % pp which is chosen uniformly in the range [0, 2pi).
                phi = (pRand * 2*pi); 
                % reconstruct timeseries with new randomized phase
                % and from this the surrogate time series is given by the inverse Fourier transform:
                sosig = real(ifft(fo .* exp(1i*phi))); srsig = real(ifft(fr .* exp(1i*phi)));
                % fisher transformed version of the correlation between group                
                oISC = atanh(corr(sosig,gcotemp)); rISC = atanh(corr(srsig,gcrtemp)); 
                % then average over hbs and store result               
                ISCNull(nc,cc,ss,nn) = (oISC + rISC)/2;


            end

        end

    end
end    
% save results
save("ISCNull.mat","ISCNull")
toc
%% see the results

% hists by condition
figure
for cc = 1:nCond
    subplot(nCond,1,cc)
    histogram(squeeze(ISCNull(cc,1,1,:)))
end

% hists by subject
figure
for ss = 1:nSub
    subplot(nSub,1,ss)
    histogram(squeeze(ISCNull(1,1,:,:)));
end

figure
% hists by channel
for cc = 1:nChan-100
    subplot(nChan-100,1,cc)
    histogram(squeeze(ISCNull(1,:,1,:)));
end


