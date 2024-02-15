%% Compute ISCs on hold out datasets

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
ISC = nan(nCond,nChan,nHb,nSub,nSub - 1);
subList = 1:nSub; 

% for each subject
for ss=1:nSub
    % first, remove a subject from list 
    tsList = subList(subList ~= ss);
    % set up matrix index
    tcount = 1;
    for ts = tsList
        gs = setdiff(tsList,ts);
        % get bad channels for the left out participants in the hold out
        % set
        BC = BadChannels(:,gs);
        % get subject and subject - 1 data for each condition
        for nc=1:nCond
            scdata = Group{nc}(:,:,:,ts); gcdata = Group{nc}(:,:,:,gs);
            
            % for each channel
            for cc = 1:nChan
              
                % fisher transformed version of the correlation between group
                o = atanh(corr(scdata(:,cc,1),mean(gcdata(:,cc,1,~BC(cc,:)),4)));
                r = atanh(corr(scdata(:,cc,2),mean(gcdata(:,cc,2,~BC(cc,:)),4)));
                ISC(nc,cc,1,ss,tcount) = o;
                ISC(nc,cc,2,ss,tcount) = r;

            end
            
        end
        tcount = tcount + 1;
    end

end
    
save("ISCHoldOut.mat",'ISC')
