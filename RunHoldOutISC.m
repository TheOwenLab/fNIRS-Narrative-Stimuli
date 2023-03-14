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

% preallocate ISC results
ISC = nan(nCond,nChannel,nSub,nSub - 1);
subList = 1:nSub; 
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
                % fisher transformed version of the correlation between group
                o = atanh(corr(scdata(:,cc,1),mean(gcdata(:,cc,1,:),4)));
                r = atanh(corr(scdata(:,cc,2),mean(gcdata(:,cc,2,:),4)));
                ISC(nc,cc,ss,tcount) = (o + r)/2;             
            end

        end
    tcount = tcount + 1;
    end
    

    
end
    
save("ISCHoldOut.mat",'ISC')
