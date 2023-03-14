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
Group = {BYD;BYDN;Taken;TakenNoise};
nIter = 1000; 
% preallocate ISC results
ISCNull = nan(nCond,nChannel,nSub,nSub - 1,nIter);
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
            gs = setdiff(tsList,ts);
            % get subject and subject - 1 data for each condition
            for nc=1:nCond
                scdata = Group{nc}(:,:,:,ts); gcdata = Group{nc}(:,:,:,gs);
                % for each channel
                for cc = 1:nChannel
                    pRand = rand(length(scdata),1);
                    % get channel data for left out and group 
                    sctemp = squeeze(scdata(:,cc,:)); gctemp = squeeze(gcdata(:,cc,:,:));
                    % grab hbo
                    scotemp = sctemp(:,1); gcotemp = mean(gctemp(:,1,:),3);
                    % grab hbr
                    scrtemp = sctemp(:,2);  gcrtemp = mean(gctemp(:,2,:),3);
                    % return phase randomzied signal for both hbs 
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
                    o = atanh(corr(sosig,gcotemp));
                    r = atanh(corr(srsig,gcrtemp));
                    ISCNull(nc,cc,ss,tcount,nn) = (o + r)/2;             
                end

            end
        tcount = tcount + 1;
        end



    end
    toc
end
save("ISCNullHoldOut.mat",'ISCNull')
