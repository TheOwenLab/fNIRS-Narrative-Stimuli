%ISC ML where participant info doesn't leak into group

% load in data
filename = "PreprocessedData.mat";
data = load(filename);
data = data.data;
% set loop parameters
nSub = length(data); nCond = 4; 
nChannel = 121; nHb = 2;

% put data in easy to use format
Rest = cat(4,data.Rest); BYD = cat(4,data.BYD); BYDN = cat(4,data.BYDN);
Taken = cat(4,data.Taken); TakenNoise = cat(4,data.TakenNoise);
% drop to odd values for the fft
Rest = Rest(1:end-1,:,:,:); BYD = BYD(1:end-1,:,:,:);
Group = {BYD;BYDN;Taken;TakenNoise};
nIter = 1000; 
% preallocate ISC results
ISCNull = nan(nCond/2,nChannel,nSub,nSub - 1,nIter);
subList = 1:nSub; 

% for each iteration
for nn = 1:nIter
    tic
    % for each subject
    for ss=1:nSub
        % first, remove a subject from list 
        tsList = subList(subList ~= ss);
        % set up matrix index
        tcount = 1;
        for ts = tsList
            counter = 0;
            gs = setdiff(tsList,ts);
            % get subject and subject - 1 data for each condition
            for nc=1:2:nCond
                counter = counter + 1;
                scidata = Group{nc}(:,:,:,ts); gcidata = Group{nc}(:,:,:,gs);
                scsdata = Group{nc+1}(:,:,:,ts); gcsdata = Group{nc+1}(:,:,:,gs);
                % for each channel
                for cc = 1:nChannel
                    pRandi = rand(length(scidata),1);
                    pRands = rand(length(scsdata),1);
                    % get channel data for left out and group 
                    scitemp = squeeze(scidata(:,cc,:)); gcitemp = squeeze(gcidata(:,cc,:,:));
                    scstemp = squeeze(scsdata(:,cc,:)); gcstemp = squeeze(gcsdata(:,cc,:,:));
                    % grab hbo for intact
                    scoitemp = scitemp(:,1); gcoitemp = mean(squeeze(gcitemp(:,1,:)),2);
                    % grab hbr for intact
                    scritemp = scitemp(:,2); gcritemp = mean(squeeze(gcitemp(:,2,:)),2);
                    % grab hbo for scrambled
                    scostemp = scstemp(:,1); gcostemp = mean(squeeze(gcstemp(:,1,:)),2);
                    % grab hbr for scrambled
                    scrstemp = scstemp(:,2); gcrstemp = mean(squeeze(gcstemp(:,2,:)),2);
                     % return phase randomzied signal for both hbs 
                    %theta = angle(z) returns the phase angle in the interval [-π,π] for each element of a complex array z. 
                    %The angles in theta are such that z = abs(z).*exp(i*theta).
                    foi = fft(scoitemp); fri = fft(scritemp);
                    fos = fft(scostemp); frs = fft(scrstemp);
                    % A "phase-randomized" Fourier transform X(f) is made by rotating the phase P at 
                    % each frequency f by an independent random variable 
                    % pp which is chosen uniformly in the range [0, 2pi).
                    phi = (pRandi * 2*pi); phis = (pRands * 2*pi); 
                    % reconstruct timeseries with new randomized phase
                    % and from this the surrogate time series is given by the inverse Fourier transform:
                    soisig = real(ifft(foi .* exp(1i*phi))); srisig = real(ifft(fri .* exp(1i*phi)));
                    sossig = real(ifft(fos .* exp(1i*phis))); srssig = real(ifft(frs .* exp(1i*phis)));
                    % fisher transformed version of the correlation between group                
                    oISCi = atanh(corr(soisig,gcoitemp)); rISCi = atanh(corr(srisig,gcritemp)); 
                    oISCs = atanh(corr(sossig,gcostemp)); rISCs = atanh(corr(srssig,gcrstemp)); 
                    % store mean difference between conditions 
                    otemp = oISCi - oISCs; rtemp = rISCi - rISCs;
                    ISCNull(counter,cc,ss,tcount,nn) = (otemp + rtemp)/2;             
                end

            end
        tcount = tcount + 1;
        end



    end
    toc
end
save("ISCNullHoldOutIS.mat",'ISCNull')
